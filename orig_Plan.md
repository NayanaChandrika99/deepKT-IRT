Alright, let’s turn this into a buildable system spec.

---

## 1. Why & for whom

**Audience:**

* Med students & residents using **UWorld Medical QBank** (Step 1, Step 2, ABIM, ABFM) and the **UWorld Medical Library**.
* They already rely heavily on **vivid illustrations, labeled diagrams, charts, tables, and clinical images** in QBank explanations and the Library.

**Problem:**

* The visuals are high-yield, but interaction is primitive: scroll / zoom / read labels.
* Students often think in **point-and-ask** mode (“What’s *this*?”, “How does *this* connect to *that*?”), not in perfect anatomy terminology.

**Goal:**
Build a **Visual Understanding Engine** for UWorld Medical Library + QBank visuals that lets learners:

* Click / tap directly on a diagram or clinical image and ask:

  * “What is this structure?”
  * “What does this connect to?”
  * “Why is this area abnormal?”
* Get **grounded, cited explanations** based only on UWorld content (Library + QBank rationales), not hallucinated medical facts.

This positions UWorld against competitors like AMBOSS, who are already shipping AI assistants grounded in their knowledge library.

---

## 2. Product concept – one-paragraph spec

> **UWorld Visual Tutor for Images**:
> A region-aware tutor layered on top of UWorld’s Medical Library and QBank visuals. Students click anywhere on an illustration, flowchart, table, or clinical image, ask a question, and get a short explanation grounded in UWorld articles and rationales. Under the hood, the system resolves which region(s) the user meant, looks up the associated structure and relations (e.g., “connects to”, “adjacent to”), retrieves the relevant explanation snippets, and has a vision-language model phrase the answer with citations and a highlighted region overlay.

---

## 3. High-level system architecture (UWorld context)

Assume a typical UWorld-like stack: **SPA front-end + .NET microservices + SQL/Databricks on Azure** (this is consistent with UWorld’s .NET full-stack hiring and common Angular + ASP.NET Core + Azure patterns).

### 3.1 Big-picture architecture

```text
+---------------------------+         +------------------------------+
|  UWorld Web / Mobile UI   |  HTTPS  |   API Gateway / Routing      |
|  (QBank + Med Library)    +---------> (Existing .NET APIs +        |
|                           |         |  NEW Visual Tutor Service)   |
+-------------+-------------+         +---------------+--------------+
              |                                           |
              |                                           v
              |                            +------------------------------+
              |                            |  Visual Tutor Service        |
              |                            |  (orchestrator)              |
              |                            |  - Click + question handler  |
              |                            |  - Region resolution         |
              |                            |  - Evidence retrieval        |
              |                            |  - VLM call + guardrails     |
              |                            +------------------------------+
              |                                           |
              |                                           v
              |       +----------------------+    +------------------------+
              |       |  AI Services Layer   |    |   Data / Content Layer |
              |       |----------------------|    |------------------------|
              |       | - Segmentation API   |    | - Image store (Blob)   |
              +------>| - Embedding service  |    | - Region library       |
                      | - VLM service        |    | - Relation graphs      |
                      |                      |    | - Vector index         |
                      +----------------------+    | - QBank + Library DB   |
                                                  +------------------------+
```

* **Visual Tutor Service**: new microservice that you own. All “smart” behavior is orchestrated here.
* **AI services** can be separate microservices or integrated:

  * Segmentation: promptable segmentation model (SAM/SAM2-like) for region masks.
  * Embedding: vision(-language) encoder for region embeddings.
  * VLM: vision-language model for phrasing explanations from evidence.

---

## 4. Offline pipeline – building the region-aware visual index

This is the backbone. You build it once, then serve queries cheaply.

### 4.1 Content sources

1. **QBank explanations** (Step 1, Step 2, ABIM, ABFM, etc.)

   * Each explanation includes: text + 0–N visual assets (illustrations, diagrams, charts, radiology, etc.).

2. **Medical Library articles**

   * 600+ peer-reviewed, specialty-organized articles with integrated visuals.

You want a **canonical visual asset table**:

```text
Table: visual_assets
-----------------------------------------------
asset_id          PK
source_type       ENUM('QBank','MedLibrary')
source_id         (question_id OR article_id)
figure_label      (e.g., "Figure 3.2", optional)
asset_type        ENUM('illustration','diagram','flowchart',
                       'table','chart','clinical_photo')
image_uri         (Azure Blob path)
topic             (cardiology, pulmonary, etc.)
system            (CV, GI, etc.)
created_at, updated_at
version
```

### 4.2 Region & mask generation (borrowed: programmatic masking)

We borrow from **Segment Anything (SAM / SAM2)**: promptable segmentation that, given a point/box, returns a mask, and can be used to auto-generate masks for many objects in an image.

**Step 1 – Auto-propose regions**

* For each `visual_asset`:

  * Run a segmentation pipeline (SAM2/MedSAM-style) offline to propose 10–100 candidate masks.
  * Filter by size/shape (ignore tiny noise regions).
  * Group masks spatially (likely structures).

**Step 2 – Human-in-the-loop labeling**

Create an internal **“Region Annotation Tool”**:

* UI loads the UWorld figure.
* Shows SAM-proposed masks as overlays.
* Annotator (MD/editor) clicks a region, chooses:

  * Canonical structure name (from anatomy ontology).
  * Optional synonyms (e.g., “LAD”, “left anterior descending artery”).
  * Structure type (artery / vein / chamber / valve / nerve / etc.).
* Saves mask + label.

**Region schema:**

```text
Table: visual_regions
-------------------------------------------------
region_id         PK
asset_id          FK -> visual_assets
mask_uri          (RLE or binary mask file)
bbox              [x_min, y_min, x_max, y_max]
canonical_label   "left_ventricle"
label_type        "structure" | "pathology" | "landmark"
color_hint        "red" | "blue" | ...
created_by        user_id
created_at, updated_at
```

This is where we **borrow** the “programmatic masks” idea from the diffusion job: masks become first-class entities we can reason about.

### 4.3 Relation graph (borrowed: multi-subject disentangling)

For high-value diagrams (heart, lungs, neuro pathways), add a **relation graph**:

```text
Table: visual_relations
-------------------------------------------------
relation_id       PK
asset_id          FK
from_region_id    FK -> visual_regions
to_region_id      FK -> visual_regions
relation_type     ENUM('branches_from',
                       'drains_to',
                       'adjacent_to',
                       'supplies',
                       'innervates')
evidence_snippet_id  FK -> explanation_snippets
notes
```

This is where we encode “what connects to what”. For questions like:

> “What is the link between this artery and that chamber?”

we answer from this graph, not from model “intuition.”

### 4.4 Text & vision embeddings

For each region & whole-asset:

1. **Region crop extraction**

   * Use `mask` or `bbox` to crop the image with a bit of padding.

2. **Text side (for tables/labels)**

   * Run OCR on region crop → `ocr_text`.
   * Generate a short caption with a general VLM (offline) → `auto_caption`.
   * Embedding: text embedding model (e.g., OpenAI text-embed or similar).

3. **Vision side**

   * Use a **vision-language encoder** (CLIP/SigLIP, or more document-focused models if you want to go fancy later) to embed:

     * Full asset → `asset_embedding`.
     * Region crop → `region_embedding`.

Store in Delta / SQL:

```text
Table: visual_region_embeddings
-------------------------------------------------
region_id        FK
vision_embedding (vector)
text_embedding   (vector)
model_version    (e.g., "siglip-2025-01")
```

And for whole images:

```text
Table: visual_asset_embeddings
-------------------------------------------------
asset_id         FK
vision_embedding (vector)
text_embedding   (vector)
model_version
```

Then create a **vector index** (Databricks Vector Search or Azure AI Search vector index) over these embeddings for fast approximate nearest neighbor search.

### 4.5 Evidence snippets (for grounding)

Link every region to at least one **UWorld-authored snippet**:

```text
Table: explanation_snippets
-------------------------------------------------
snippet_id       PK
source_type      ('QBank','MedLibrary')
source_id        (question_id or article_id)
start_offset     int
end_offset       int
snippet_text     TEXT
```

Link them:

```text
Table: region_evidence
-------------------------------------------------
region_id        FK -> visual_regions
snippet_id       FK -> explanation_snippets
evidence_type    ('definition','function','connection','pathology')
```

This is where the **VLM is constrained**: it must answer using these snippets as its factual basis.

---

## 5. Online flow – when a student clicks & asks

Now the fun part: run-time behavior.

### 5.1 High-level interaction loop

```text
[Browser: UWorld UI]
   |
   | 1. Click (x,y) on figure + question text
   v
[Visual Tutor API]
   |
   | 2. Region resolution & disambiguation
   v
[Region Engine]
   |
   | 3. Evidence retrieval (snippets + relations)
   v
[Retrieval Layer] --(DB / vectors)--> [Data Layer]
   |
   | 4. Build "evidence pack"
   v
[VLM Service]
   |
   | 5. Grounded answer + highlights
   v
[Browser: overlay + explanation + citations]
```

### 5.2 Step-by-step

**Step 1 – Client → Visual Tutor API**

The front-end (Angular / whatever) already has:

* `asset_id` (which figure the user is viewing)
* Click coordinates in image space: `(x, y)`
* Optional question text: `"What is this?"` / `"How does this connect to that?"`

Payload:

```json
POST /visual-tutor/query
{
  "asset_id": "fig_heart_001",
  "clicks": [
    {"x": 420, "y": 310}
  ],
  "question": "What is this red area beside the main artery?"
}
```

**Step 2 – Region resolution (borrowed: handling bleed)**

The **Region Engine** does:

1. Fetch all `visual_regions` for `asset_id`.
2. For each region, compute a **score**:

   * `inside_mask_score` (is click deep inside the mask or near edge?)
   * `distance_to_centroid`
   * Optional: text match between question and region labels (if “artery” appears).
3. Rank candidate regions. If top-1 >> top-2 → pick it.
   If scores are close → *ambiguous mode* (like multi-subject bleed):

   * Either ask a follow-up:

     * “Did you mean [LAD] or [LCX]? (shows both highlighted)”
   * Or refuse to answer directly.

At the end you have:

```json
"resolved_regions": [
  { "region_id": "r_left_ventricle", "score": 0.91 }
]
```

**Step 3 – Retrieve evidence**

For each `resolved_region`:

* Fetch `region_evidence` → list of `snippet_id`s
* Fetch `visual_relations` where `from_region_id` or `to_region_id` matches (for “connects to” questions)
* Optionally run a **small embedding search** to pull a few extra related snippets for disambiguation.

Build an **evidence pack**:

```json
{
  "asset": {
    "asset_id": "fig_heart_001",
    "image_uri": "https://blob/fig_heart_001.png"
  },
  "regions": [
    {
      "region_id": "r_left_ventricle",
      "canonical_label": "Left Ventricle",
      "mask_uri": "mask_r_left_ventricle.rle",
      "relations": [
        {
          "relation_type": "supplies",
          "to_region_label": "Systemic Circulation"
        }
      ],
      "evidence_snippets": [
        "The left ventricle pumps oxygenated blood into the aorta...",
        "Hypertrophy of the left ventricle is commonly seen in..."
      ]
    }
  ]
}
```

**Step 4 – Call VLM with constraints**

Prompt pattern:

> “You are a UWorld medical tutor. You must answer using ONLY the provided evidence from UWorld content. If the evidence is insufficient, say you don’t know and suggest reading the linked explanation or article.”

Pass:

* The original image (or region crop with a highlight)
* The question
* Structured metadata (canonical label, relations)
* The snippet texts

The VLM **cannot browse the internet**; it just rewrites / summarizes UWorld text and points to the region.

**Step 5 – Return to client**

Response:

* Markdown/text explanation
* List of citations: `{snippet_id: …}` → front-end maps them to “QBank explanation” / “Medical Library article” links.
* Highlight mask(s) to overlay on the figure.

---

## 6. Infra & latency (borrowing from the “visual conversation engine”)

From that diffusion job spec we borrow the mindset:

* **Sub-1 second interaction budget**

  * Region lookup & scoring: in-memory / cached → < 100 ms
  * Evidence retrieval: one DB + one vector query → < 150 ms
  * VLM answer: quantized / small vision model, warm on GPU → 500–800 ms
* **Serverless GPU** for:

  * VLM inference
  * Segmentation API (for dynamic cases / future features)
* **Precompute everything** you can:

  * Region masks & labels
  * Embeddings
  * Relation graphs
  * Evidence snippet links

Deployment on Azure could look like:

```text
+---------------------------+
| Azure Front Door / CDN    |
+-------------+-------------+
              |
              v
+---------------------------+
| App Service / AKS         |  (API Gateway + Visual Tutor)
+-------------+-------------+
              |
              +------> [Databricks / SQL]  (tables, vector index) 
              |
              +------> [GPU Inference (Container Apps / Modal-like)]
                           - VLM
                           - (Optional) Segmentation service
```

---

## 7. Safety, evaluation, and logging

**Safety rules:**

1. **No anatomy invention**

   * If region not confidently resolved → disambiguate or refuse.
   * If no evidence snippets → answer: “This diagram does not provide enough information; refer to [link]”.

2. **Always show evidence**

   * Highlight region.
   * Show snippet excerpts with “Read full explanation” / “Open Medical Library article”.

3. **Logging:**

   * Query (anonymized) + click coords
   * Resolved region(s) + scores
   * Evidence snippet IDs
   * VLM output
   * User feedback (“Helpful?”, “Wrong structure?”)

4. **Evaluation:**

   * **Region accuracy**: does the system pick the correct structure? (label-level)
   * **Relation accuracy**: for “connects to” questions, do edges match the curated graph?
   * **Groundedness**: internal eval on whether each claim is supported by evidence.

---

## 8. Build order (what you do first)

If you want something you can actually start coding, I’d do it in this order:

1. **MVP region library for a small set of figures**

   * Pick: 10–20 core diagrams (heart, lung, kidney).
   * Manually define:

     * `visual_assets`, `visual_regions`, `visual_relations`, `explanation_snippets`, `region_evidence`.
   * Skip segmentation models at first → use simple bounding boxes drawn by annotators.

2. **Simple Region Engine**

   * Given `(asset_id, x, y)`:

     * Resolve nearest region (by bbox, then refine later).
     * Log ambiguous clicks.

3. **Evidence retrieval**

   * Hard-link each region to 1–3 explanation snippets.
   * Build a trivial snippet API that returns those texts.

4. **VLM integration**

   * Use a hosted multimodal model.
   * Prompt: “Use only this text + this region; do not add new facts.”
   * Add “refuse if insufficient evidence.”

5. **Front-end integration**

   * Add a “Ask about this figure” mode in UWorld-like UI.
   * Implement click → highlighted region → side-panel answer with citations.

6. **Then iterate**

   * Replace manual bboxes with SAM/SAM2-assisted masks.
   * Add relation graph for a couple of diagrams.
   * Add overlap scoring / disambiguation UI.
   * Move embeddings + vector search into Databricks / Azure AI Search.
