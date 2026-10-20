"""No-shot moralization prompt."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert in moral foundations theory and the analysis of "
    "political rhetoric. Your task is to rate the degree of MORALIZATION "
    "in a speech delivered in the Swedish parliament (Riksdagen)."
)

USER_PROMPT_TEMPLATE = """Analyze the following Swedish parliamentary speech for the presence of moral framing.

The unit of analysis is the speech as a whole. Rate it as a single rhetorical act, weighing both how much of the speech deploys moral framing and how central that framing is to the speaker's argument.

# Construct: moral framing

Moral framing is language that evaluates political objects (policies, actors, groups, events) by appeal to principles of right and wrong rather than by appeal to consequences, costs, feasibility, or empirical claims about what works. Moral framing typically draws on one or more moral foundations: care versus harm, fairness versus cheating, loyalty versus betrayal, authority versus subversion, sanctity versus degradation, and liberty versus oppression. It signals that the issue at stake concerns fundamental values rather than tractable policy trade-offs.

# What counts as moral framing

Look for:
- Appeals to moral foundations, for example, framing a policy as a matter of fairness, harm prevention, dignity, betrayal of principle, or violation of sacred values.
- Evaluation grounded in obligation, duty, rights, justice, or principle rather than in efficiency, feasibility, or evidence.
- Rhetorical treatment of an issue as one on which compromise would be wrong rather than merely unwise.
- Invocation of universal or near-universal principles ("every child deserves," "no human being should").
- Framing of political choices in absolute terms of right and wrong.

# Implicit moral framing

Moral framing is often indirect, particularly in Swedish parliamentary speech. Attend to:
- Rhetorical questions that presuppose a shared moral standard.
- Appeals to national, collective, or personal character that imply moral failing.
- Analogical or historical framing that places a current policy or actor near a recognized moral wrong.
- Contrast structures that juxtapose the suffering or sacrifice of one group with the gain or indifference of another.
- Implicit invocation of sacred or constitutive values.

Sarcasm and irony count as moral framing ONLY when the literal meaning the sarcasm inverts is itself a moral claim. Apply this decoding test: state the proposition the sarcasm asserts. If that proposition makes a moral claim about harm, fairness, dignity, or duty, the sarcasm is moralizing. If it merely asserts that policy is bad, incompetent, or absurd, it is not.

# What does not count as moral framing

The following are conceptually distinct from moral framing. They should not raise the rating on their own, even when they are intense or pervasive:
- Negative sentiment, anger, frustration, or emotional intensity.
- Sarcasm, ridicule, or dismissive rhetoric, except where it functions as implicit moral framing (see above).
- Personal attacks, insults, or hostility toward political opponents.
- In-group praise or out-group derogation expressed without moral content.
- Procedural complaints or accusations of incompetence.
- Strong language about consequences ("disaster," "catastrophe") when the underlying argument is pragmatic.

A speech can be hostile without being morally framed: criticizing an opponent as ineffective, lazy, or out of touch is not moral framing. A speech can also be morally framed without being hostile: calmly arguing that a policy is unjust on principled grounds is moral framing. Rate only the moral framing component, even when other features are salient.

# Reasoning

Within <thinking></thinking> tags, work through the speech step by step. Identify passages that appeal to moral foundations or principled evaluation. Separately identify passages that are merely critical, hostile, sarcastic, or pragmatically negative, and exclude these from the moral framing assessment. Then consider how central moral framing is to the speech as a whole.

# Rating scale

Assign an integer from 1 to 5 within <answer></answer> tags:

1 = No moral framing. The speech is descriptive, technical, or pragmatic. Evaluation, where present, rests on consequences, costs, evidence, or feasibility. Speeches at this level may still be critical or hostile, but their criticism is not morally framed.

2 = Minimal moral framing. The speech is predominantly pragmatic but contains occasional value-laden language or implicit moral assumptions. Moral framing is peripheral to the argument.

3 = Moderate moral framing. The speech mixes pragmatic and moral arguments. At least one section invokes moral principles or foundations explicitly, but the speech also rests substantially on practical considerations.

4 = Substantial moral framing. The speech argues largely from moral principle. Policies and actors are evaluated primarily by reference to right and wrong, fairness, harm, or duty. Pragmatic considerations are subordinated to moral ones.

5 = Pervasive moral framing. The speech is built almost entirely around moral evaluation. Issues are presented as fundamental questions of right and wrong, and compromise is treated as principally unacceptable. The argument rests on appeals to absolute or near-absolute principles.

# Speech to rate

<speech>
{speech}
</speech>

Provide your reasoning within <thinking></thinking> tags, then the integer rating within <answer></answer> tags."""
