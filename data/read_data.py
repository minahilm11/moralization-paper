
from lxml import etree
import pandas as pd
import re
import os
from time import time
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import List, Optional
from multiprocessing import Pool, Manager, cpu_count
import tiktoken
from collections import defaultdict



@dataclass
class Session:
    """
    Args:
    directory: directory of the session xml-document (also identificator for the exact session.)
    year: year of the session
    protocols: List of protocol objects (Protocol) within the session
    """
    directory: str
    year: int
    protocols: Optional[list]

@dataclass
class Party:
    """
    Args:
    id: party id that links to other objects
    name: name of the party
    id_standard: standard party id used outside of the database
    dissolution: date/year the party's dissolution
    """
    id:str
    name:str
    id_standard:Optional[str]
    dissolution:Optional[str]

@dataclass
class Speech:
    """
    Args:
    speech_idx: the index of the speech within the protocol section
    speech_id: unique identifier for the speech of format 'xml-filename_section-id_speech-index'
    speaker: string value within the protocol for the speaker
    who: speaker id that links to other data objects
    speech_date: date of the speech
    text: the speech
    party: party object for the party of the politician holding the speech if exists
    """
    speech_idx: int
    speech_id: str
    speaker: Optional[str]
    who: Optional[str]
    speech_date: Optional[date]
    text: str
    party: Optional[Party] = None

    def __post_init__(self):
        def parse(d):
            # parse the date variable
            if isinstance(d, str):
                try:
                    return datetime.strptime(d, "%Y-%m-%d").date()
                except ValueError:
                    return None
            return d
        self.speech_date = parse(self.speech_date)


@dataclass
class Protocol:
    """
    Args:
    section_id: id for the section of the protocol
    section type: debateSection if speeches
    protocol_date: date of the protocol
    title: title of the section - tells about the content of the debate etc.
    speeches: list of speeches held in the protocol section as Speech dataclass objects
    """
    section_id: str
    section_type: str
    protocol_date: Optional[date]
    title: Optional[str]
    speeches: List[Speech] = field(default_factory=list)

    def __post_init__(self):
        def parse(d):
            # parse the date variable
            if isinstance(d, str):
                try:
                    return datetime.strptime(d, "%Y-%m-%d").date()
                except ValueError:
                    return None
            return d
        self.protocol_date = parse(self.protocol_date)



@dataclass
class PartyAffiliation:
    """
    Args:
    start_date: date the person started in the party
    end_date: date the person left the party
    party: Party dataclass object
    """
    start_date: Optional[date]
    end_date: Optional[date]
    party:Party

    def __post_init__(self):
        def parse(d):
            if isinstance(d, str):
                if len(d) == 4 and d.isdigit():
                    return date(int(d), 1, 1)
                try:
                    return datetime.strptime(d, "%Y-%m-%d").date()
                except ValueError:
                    return None
            return None
        self.start_date = parse(self.start_date)
        self.end_date = parse(self.end_date)


@dataclass
class Representative:
    """
    Args:
    id: id for the person that connects to Speeches etc.
    name: name of the representative
    affiliations: all PartyAffiliations of the representative as a list
    speeches: all Speeches as a list
    """
    #TODO add ministership and other affiliations
    id:str
    name:str
    affiliations: List[PartyAffiliation] = field(default_factory=list)
    speeches:List[Speech] = field(default_factory=list)

    def get_party_on(self, query_date: date) -> Party:
        for a in self.affiliations:
            try:
                if a.start_date <= query_date and (a.end_date is None or query_date <= a.end_date):
                    return a.party
            except TypeError:
                # if no party affiliation
                continue
        # if did not find a party within the range, return a party with no start or end dates define
        #print(self.affiliations)
        skip = ["", None]
        leftover = [a for a in self.affiliations if a.end_date in skip and a.start_date in skip]
        if len(leftover) > 0:
            return leftover[0].party
        #print(self.affiliations)
        return Party(id='unknown', name='no party', id_standard=None, dissolution=None)



def parse_sessions(start_year: int, end_year:int, root_dir) -> list:
    """Walks the given root finds files in subfolders that have a year in a given range.
    """
    # TODO save commentSection comments

    file_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        # find all 4-digit numbers that look like years
        try:
            year = int(dirpath.split(os.sep)[-1][:4])
        except ValueError:
            continue
        # keep only if any year is in the desired range
        if start_year <= year <= end_year:
            for fname in filenames:
                file_paths.append((os.path.join(dirpath, fname), year))
    return file_paths



def parse_protocols(args) -> Session:
    """Parses an xml file with speeches by dividing the protocol to sections and then speeches.
    Saves both as dataclass objects.
    """
    xml_file, year = args
    tree = etree.parse(xml_file)
    root = tree.getroot()

    # Namespace dictionary
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    protocols = []
    date_el = root.find('.//tei:docDate', ns)
    doc_date = date_el.attrib.get('when') if date_el is not None else None

    # Loop through each protocol section
    for div in root.xpath('.//tei:div[not(@type="preface")]', namespaces=ns):
        section_id = div.attrib.get('{http://www.w3.org/XML/1998/namespace}id')
        section_type = div.attrib.get('type', '')

        # Extract optional title
        title_node = div.find('tei:note[@type="title"]', namespaces=ns)
        title = title_node.text.strip() if title_node is not None and title_node.text else None

        speeches = []
        current_speaker = None
        current_who = None
        idx = 0

        # Iterate over child nodes
        for node in div.iter():
            tag = etree.QName(node).localname  # Get tag without namespace

            if tag == "note" and node.attrib.get("type") == "speaker":
                if current_speaker:
                    #print("adding another speech")
                    # add last speaker's speech as dataclass
                    try:
                        speech_id = f"{xml_file}_{section_id}_{idx}"
                        speeches.append(Speech(
                            speech_idx=idx,
                            speech_id=speech_id,
                            speaker=current_speaker,
                            speech_date=doc_date,
                            who=who,
                            text=text
                        ))
                        text = ""
                        idx +=1
                    except UnboundLocalError:
                        pass
                # find current speaker
                #print("New speaker")
                current_speaker = node.text if node.text else None
                if current_speaker:
                    current_speaker = clean_text(node.text)

            elif tag == "u":
                #speech_id = node.attrib.get('{http://www.w3.org/XML/1998/namespace}id')
                who = node.attrib.get("who")
                if who == current_who:
                    text = " ".join((text, clean_text(" ".join(seg.text for seg in node.findall('tei:seg', namespaces=ns) if seg.text))))
                else:
                    current_who = who
                    # Combine all <seg> text inside this <u>
                    text = clean_text(" ".join(seg.text for seg in node.findall('tei:seg', namespaces=ns) if seg.text))

        #print("adding the last speech")
        # add last speech and protocol if exists
        if current_speaker and current_who:
            speech_id = f"{xml_file}_{section_id}_{idx}"
            speeches.append(Speech(
                speech_idx=idx,
                speech_id=speech_id,
                speaker=current_speaker,
                speech_date = doc_date,
                who=who,
                text=text
                    ))

        # if protocol contains speeches
            protocols.append(Protocol(
                section_id=section_id,
                section_type=section_type,
                protocol_date=doc_date,
                title=title,
                speeches=speeches
            ))

    return Session(directory=xml_file, year=year, protocols=protocols)

def clean_text(text):
    if not text:
        return ""
    # Strip each line, remove empty lines, join with space
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Join lines and collapse multiple spaces
    return re.sub(r'\s+', ' ', " ".join(lines))

def generate_protocol_MP(sessions, num_workers = cpu_count()-1):
    """Parses the protocol xml-documents in parallel.
    """
    with Pool(processes=num_workers) as pool:
        # imap_unordered allows streaming results, preventing RAM overload
        for session in pool.imap_unordered(parse_protocols, sessions, chunksize=1):
            if session:
                yield session

def add_speeches(reps: list[Representative], speeches: list[Speech]):
    """Add the speeches to the rep as a list of Speech class objects.
    """
    speech_dict = defaultdict(list)
    for speech in speeches:
        speech_dict[speech.who].append(speech)

    # Assign speeches to reps
    for rep in reps:
        rep.speeches = speech_dict.get(rep.id, [])


def add_party_to_speeches(all_speeches: list[Speech], speaker_lookup: dict):
    """Add party attribute to all speeches.
    """
    for sp in all_speeches:
        speaker = speaker_lookup.get(sp.who)
        if speaker:
            sp.party = speaker.get_party_on(sp.speech_date)


def split_long_sequences(all_speeches: list, max_tokens = 8192):
    # Initialize tokenizer (or any tokenizer compatible with your model)
    new_speeches = []
    tokenizer = tiktoken.get_encoding("cl100k_base")
    counter = 0
        # Tokenize to count tokens
    for speech in all_speeches:
        text = speech.text
        tokens = tokenizer.encode(text)
        # If text fits, return as-is
        if len(tokens) <= max_tokens:
            new_speeches.append(speech)
            continue

        # drop speeches longer than 14 000 tokens
        elif len(tokens) > 14_000:
            counter += 1
            continue

        print(len(tokens))
        counter += 1
        part1 = speech
        part2 = speech
        # Find midpoint in characters (not tokens, for simplicity)
        midpoint = len(text) // 2

        # Find nearest sentence boundary (". ") near midpoint
        split_index = text.find(". ", midpoint)
        if split_index == -1:  # fallback if not found after midpoint
            split_index = text.rfind(". ", 0, midpoint)

        # If no period-space found at all, just split at midpoint
        if split_index == -1:
            split_index = midpoint

        part1.text = text[:split_index + 1].strip()
        part2.text = text[split_index + 1:].strip()

        new_speeches.append(part1)
        new_speeches.append(part2)

    print(f"{counter} speeches were above 8192 tokens")
    print(len(new_speeches))
    return new_speeches