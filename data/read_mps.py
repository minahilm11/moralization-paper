import pandas as pd
from read_data import Representative, PartyAffiliation, Party


def read_parties(filedir:str) -> pd.DataFrame:
    """Read dataset that connects representative's to parties.
    """
    df = pd.read_csv(filedir, dtype = str)
    df = df[df['swerik_party_id'].notna()]
    return df

def read_names(filedir1: str, filedir2: str, lookup: dict) -> list[Representative]:
    """Reads a dataset of parliamentarians. Creates a dataclass object for each rep.
    """
    df = pd.read_csv(filedir1, dtype = str)
    df = df[df['primary_name']== "True"]
    party_aff = read_parties(filedir2)
    people = []
    for _, row in df.iterrows():
        affiliations = party_aff[party_aff['person_id'] == row['person_id']]
        people.append(
            Representative(
                id=row['person_id'],
                name=row['name'],
                affiliations=[
                    PartyAffiliation(
                        start_date=aff['start'],
                        end_date=aff['end'],
                        party=lookup[aff['swerik_party_id']]
                    )
                    for _, aff, in affiliations.iterrows()
                ]
            )
        )
    # adding unknown
    people.append(
            Representative(
                id="unknown",
                name='unknown',
                affiliations=[]
            )
        )
    return people

def read_party_names(filedir: str) -> list[Party]:
    """Read dataset that connects party name to id. Creates a dataclass object for each party.
    """
    df = pd.read_csv(filedir, dtype = str)

    parties = [
        Party(
            id = row['swerik_party_id'],
            name = row['party'],
            id_standard = row['party_id'],
            dissolution = row['dissolution'],
        )
        for _,row in df.iterrows()
    ]
    return parties
