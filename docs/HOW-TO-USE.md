# How to use this repo

This repo has **one rule: the chapter is the front door.** Everything else follows from it.

## The two journeys

### 📖 Reading the book, hand by hand
1. Read a chapter in the book.
2. Open the matching `chapters/chNN-*/` folder.
3. Run the numbered notebooks top-to-bottom — each narrates one technique and runs live.
4. Run `reproduce.py` to see the chapter's headline comparison on the golden set (quality **and** cost).

You never leave the chapter folder. Everything that chapter needs is in it, with links out to the
production code and the book section.

### 🛠️ Building your own system
1. `import ragkit` and pull the component you need — every default is the book's verdict pick.
2. Read [`../capstone/`](../capstone/) for the full assembled enterprise RAG system.
3. Swap any provider with one config line ([PROVIDER-SWAP.md](PROVIDER-SWAP.md)).

The walkthrough notebooks are *thin* — they import `ragkit` and narrate. The code you learn from
is the code you ship. They never diverge.

## Anatomy of a chapter folder

Every `chapters/chNN-*/` is identical in shape, so once you know one, you know all 15:

```
chNN-name/
├── README.md          # the chapter mirror: techniques, verdict, run commands,
│                      #   links to BOTH the book section AND the ragkit module
├── 01_<technique>.ipynb   # walkthroughs, numbered in the book's reading order
├── 02_<technique>.ipynb   #   (the "key" ones are also exported as .py scripts)
├── ...
└── reproduce.py       # the chapter's head-to-head experiment → prints quality + cost
```

## Anatomy of the library (`ragkit/`)

Organized by the book's four Parts, plus `core/` (shared contracts) and `eval/` (the harness):

```
ragkit/
├── core/           config · chunk schema · provider registry
├── ingestion/      parsing · chunking · embedding · indexing        (Ch 2–5)
├── retrieval/      hybrid · query · routing · rerank · context       (Ch 6–8)
├── architectures/  graph · hierarchical · adaptive · agentic · multimodal (Ch 9–12)
├── production/     generation · security · serving · observability   (Ch 13,15,16)
└── eval/           metrics · golden-set runner · judge calibration   (Ch 14)
```

A chapter folder is *where you learn it*; the matching `ragkit` module is *what you ship*. The
chapter README always links to its module, and the module docstring always cites its chapter.

## Phase status

- **Phase 1 (now):** the full navigable structure + a README/spec in every folder. Walk it,
  confirm the layout works for you.
- **Phase 2 (next):** the runnable notebooks, the `ragkit` implementation, and the capstone —
  each verified to run top-to-bottom on the all-open stack.
