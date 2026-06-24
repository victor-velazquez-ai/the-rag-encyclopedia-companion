"""Chapter 15 reproduction — retrieval-time ACL trimming and injection defense-in-depth.

Runs the chapter's central claims on the golden set:

  * retrieval-time ACL trimming — run the same query as an entitled and an unentitled caller,
    confirm the unentitled caller's chunks never enter the prompt, and measure the recall cost of
    the security pre-filter (the Chapter 5 filtered-ANN hazard, made visible), and
  * indirect-injection defense-in-depth — fire a poisoned retrieved document at the pipeline with
    the layers off vs. on, reporting attack-success rate (ASR) as each layer is added (the
    AgentDojo lesson: tool filtering alone cut targeted ASR 53.1% -> 7.5%).

Prints a security delta (ASR, leaked-chunk count) AND a quality/cost delta (recall under the
filter, added latency), because hardening that destroys retrieval is not hardening you can ship.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
