"""
Shared job-market calibration.

One block of guidance about how to read a job posting, imported by both
the judge (agents/judge.py) and the search agent (agents/search_agent.py)
so they apply the SAME standard. When these two disagree about what
counts as a match, the pipeline burns cycles: the search agent hands up a
pick it considers strong, the judge rejects it on a requirement the search
agent never weighted, and three cycles later nothing has converged.

The content is deliberately opinionated. Postings are wish lists written
by committee, and a model reading one cold will treat every bullet as
load-bearing — which produces both of the failure modes this project has
hit: a search agent that discards good matches over a missing tool name,
and a judge that approves a weak pick because it matched a lot of
keywords.

Kept separate from the prompts that use it so there is exactly one place
to correct the calibration as the market shifts.
"""

MARKET_CALIBRATION = """JOB MARKET CALIBRATION \u2014 how to weigh a posting's \
requirements.

A posting is a wish list, not a specification. A large share of any \
posting's stated requirements are aspirational, copied from a previous \
req, or added by someone who will not interview the candidate. Weigh them \
accordingly.

HARD blockers. A genuine mismatch on any of these disqualifies the pick \
outright, however well the rest lines up:
- Work authorization: the role requires citizenship, a clearance, or \
sponsorship the candidate does not have and the employer will not provide.
- A licence, registration, or accreditation legally required to do the work.
- A degree level stated as a bar the candidate does not hold (e.g. "PhD \
required" on a research-scientist role).
- Location: strictly on-site somewhere the candidate has not said they can \
be, or in another country.
- Discipline: the role's actual day-to-day is a different job \
(a pure data-engineering or pure front-end role for an ML scientist; a \
people-management role for an individual contributor).
- Seniority two or more bands off in either direction \u2014 a staff, principal, \
or director role for a mid-level candidate; a new-grad role for someone with \
eight years.

SOFT. Routinely over-specified. Do NOT treat these as real mismatches, and \
never let one hold back an otherwise good pick or resume:
- Years-of-experience thresholds. Employers inflate these as a filter. \
Within roughly two years of the stated number is a match \u2014 "5+ years" with \
four years is not a gap worth raising.
- Anything under "nice to have", "bonus", or "preferred". The employer has \
already told you it is optional.
- A named tool where the candidate has a direct equivalent: PyTorch vs \
TensorFlow, Azure vs AWS vs GCP, Postgres vs MySQL, Airflow vs Prefect. \
Tools in the same family transfer in days, and every hiring manager knows it.
- Industry or domain familiarity, for most technical roles. Someone who has \
done the method in medical imaging can do it in fintech. Treat domain as a \
tiebreaker, never a requirement, unless the posting is explicitly built \
around regulated domain knowledge.
- Ubiquitous engineering boilerplate that appears in nearly every posting \
regardless of the actual job: CI/CD, Docker, Kubernetes, agile, code review, \
observability, unit testing, "excellent communication skills".
- Stack specifics that someone with the underlying skill picks up in weeks.

WEIGHT, instead, what the job is actually about: the core discipline, the \
primary methods the role runs on day to day, seniority within one band, and \
whether the candidate has demonstrably done work of the same kind and at a \
comparable scale.

The test: a posting that matches on the core work and misses five soft items \
is a STRONG match. A posting that matches twenty keywords while the core \
work is something the candidate has never done is a WEAK one, no matter how \
much surface overlap the text shows. Keyword overlap is the most common way \
to get this wrong in both directions."""
