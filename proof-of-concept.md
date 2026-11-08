# Photo Data Provenance Monitoring

## The idea

We sell to companies that distribute photos for AI training. Before those companies deliver photos to their customers, we embed an invisible watermark, or imprint, in each photo. They can then sell and deliver the photos through their usual channels.

Our system monitors newly trained image models as they are published or updated on Hugging Face. It works in two stages. First, it ranks models by how relevant their domain appears to each client's photo collection. Then, for relevant candidates, it identifies the base model and compares its statistics with the fine-tuned model's statistics, testing for the imprint created by the client's watermarked photos. The test asks whether the observed correlation and model change could plausibly occur without training on those photos.

When a model is relevant to the client's photo domain and the statistical test indicates that the observed correlation is unlikely without the client's data, we notify the data provider and give its team evidence to review. The provider’s legal team decides whether the model’s use was authorized and whether to contact the model maker, seek a license, or pursue legal action.

## Customer

The customer is a **photo data distributor** that:

- distributes photos to AI developers or other downstream customers;
- can authorize watermarking before delivery, including any necessary contributor permissions;
- wants visibility into whether its photos may have been used in public fine-tuned models; and
- has a licensing, compliance, or legal team that can review potential matches.

### Initial prospects to validate

These are prospects, not confirmed buyers. The key qualification is whether the distributor can authorize watermarking of its images and has an interest in monitoring public model releases.

1. **Wavebreak Media** — visual dataset provider with image collections for AI training. [AI datasets](https://wavebreakmedia.com/ai-training-datasets/) · [Contact](https://wavebreakmedia.com/contact/)
2. **vAIsual** — visual AI dataset marketplace with image datasets and partner-supplied content. [Products](https://vaisual.com/products/) · [FAQ](https://vaisual.com/faq/)
3. **Depositphotos** — licenses image datasets for AI training and has a data-provider program. [AI training data](https://depositphotos.com/ai-training-data.html) · [Data providers](https://depositphotos.com/ai-training-data/suppliers.html)
4. **Alamy** — licenses a large image collection for AI training. [AI training collection](https://www.alamy.com/ai-training-media/)
5. **Shutterstock** — a larger later-stage target with established AI data licensing. [AI data licensing](https://www.shutterstock.com/data-licensing)

## Product workflow

1. **Register a provider and its photo collections.** The provider identifies the collections and licensing batches it wants monitored.
2. **Mark the photos.** Our service embeds an invisible watermark in photos before the provider delivers them. We preserve a record linking each watermark to the provider, collection, batch, and encoding date.
3. **Deliver as usual.** The provider continues selling and delivering its photos under its normal agreements and workflow.
4. **Discover and rank newly trained models.** Our crawler monitors public Hugging Face model repositories for newly published or updated fine-tuned image models. It ranks them by relevance to each client's photo-data domain, using available model descriptions, tags, examples, and other public information. This relevance ranking does not use the watermark test.
5. **Identify the base model.** For high-relevance candidates, the system identifies the stated or most likely base model from repository metadata and, where needed, model comparison.
6. **Test for the watermark imprint.** The system compares the base model's statistics with the fine-tuned model's statistics and tests whether their correlation with the client's watermark imprint could plausibly arise without training on the client's marked photos. It records the result against a validated statistical threshold.
7. **Notify the provider.** If a domain-relevant candidate's result indicates that the observed imprint is statistically unlikely without the client's data, we send an alert and evidence report identifying the model, revision, likely base, relevant photo collection, and statistical result.
8. **Provider reviews and acts.** The provider’s legal team checks the report against the relevant license, permissions, and other facts, then decides whether to contact the model developer or pursue another remedy.

Hugging Face provides documented APIs for listing model repositories and accessing model files, and documents rate limits for API use. The crawler should use supported access methods, respect repository access controls and platform terms, and only analyze models it can lawfully access. [Hugging Face Hub API](https://huggingface.co/docs/hub/en/api) · [Model downloads](https://huggingface.co/docs/hub/models-downloading)

## Technical basis and claims

Meta’s **Pixel Seal** research, published December 18, 2025, is a potential basis for invisible, robust image watermarking. The initial product scope is photos. [Meta Pixel Seal paper](https://ai.meta.com/research/publications/pixel-seal-adversarial-only-training-for-invisible-image-and-video-watermarking/)

The model-scanning and statistical comparison capability is a separate part of the product and must be built and validated. The relevance ranking is the first filter; it selects newly published models whose domain is pertinent to a client's photos. The second-stage test compares a selected model with its base model and tests whether the correlation associated with the client's watermark imprint could have occurred without training on the marked photos.

The product should report a **statistical likelihood or confidence under stated test conditions**. In product language, a flagged result means the observed model change is statistically unlikely without the client's data; it should not be presented as absolute proof or as literally impossible under every alternative explanation. Thresholds must be calibrated against false positives and false negatives before alerts are used to support legal decisions.

## Value to the provider

- Find public fine-tuned models that may have learned from the provider’s marked photos.
- Prioritize possible matches instead of manually inspecting every model release.
- Give licensing and legal teams a technical lead with a reproducible statistical report.
- Add a monitoring service to existing photo licensing relationships.

The product finds leads and produces evidence for review. It does not determine whether the use was unauthorized, whether a defense such as fair use applies, or whether a legal claim will succeed.

## Proposed pricing

These are proposed launch prices in USD, not published market benchmarks. Visual dataset providers generally quote AI dataset deals based on scope and requirements. [Depositphotos](https://depositphotos.com/ai-training-data.html) · [vAIsual](https://vaisual.com/request-a-quote/)

### Production pricing

| Component | Price |
|---|---:|
| Monitoring platform, crawler, ranking, dashboard, and reports | $1,000/month |
| Photo encoding, first 10 million photos per year | $1.00 per 1,000 photos |
| Photo encoding, above 10 million photos per year | $0.50 per 1,000 photos |
| Candidate model revisions screened | First 100/month included; $250 per additional 100 |
| Detailed candidate investigation and evidence report | $750 per candidate model revision and watermark group |
| Retest after a candidate model is updated | $300 |
| Annual minimum spend | $24,000, including platform and usage charges |

**Detailed investigation definition:** take one candidate model revision that passed domain-relevance ranking, compare it with its likely base model for one registered photo watermark group, run the calibrated statistical test, and deliver one evidence report. A different model revision or watermark group is a separate investigation.

**Example:** 6 million photos encoded, 8 detailed investigations, and 12 months of platform access totals **$24,000**: $6,000 encoding + $6,000 investigations + $12,000 platform. The 100 monthly candidate screens are included in the platform fee.

## First customer discovery

Before scaling the crawler, interview three people responsible for AI data licensing, photo dataset operations, or rights enforcement at prospective distributors. Test these questions:

- Can they authorize watermarking the photos they distribute?
- Can they add marking at a point in their existing workflow without harming delivery or image quality?
- Which public model releases would they want monitored, and how often?
- What evidence would their legal team need before contacting a model developer?
- Would they pay for monitoring, per-photo encoding, candidate investigation, or a combination?

## Assumptions and risks to validate

- **Watermark persistence:** Resizing, compression, cropping, augmentation, or other processing may weaken or remove a watermark.
- **Model-detection validity:** The imprint may not persist in detectable form across different image-model architectures, training methods, or amounts of fine-tuning.
- **Base-model identification:** A model repository may omit or misstate its base model. Weight comparison may be needed, and some candidates may not have a clear parent model.
- **Public-model coverage:** Hugging Face scanning covers models available there and accessible to the crawler. It will not reveal private models or models distributed elsewhere.
- **Compute and scale:** Downloading and comparing model checkpoints can be expensive. Candidate ranking should limit the number of detailed investigations.
- **Rights to modify:** The provider must have authority to apply the mark to the photos, including any required contributor, model, or property permissions.
- **Legal interpretation:** A watermark match is a lead for investigation, not a conclusion that a license was breached or that the use lacks a legal defense.
- **Customer value:** Providers must care enough about public-model monitoring to pay recurring fees and cover encoding and investigation usage.

## Pilot success criteria

A pilot succeeds if the provider can mark a real photo collection in its existing delivery workflow; the crawler finds and ranks relevant public fine-tuned image models; controlled tests distinguish models trained with marked photos from appropriate negative controls at an agreed error rate; and the provider’s legal or licensing team finds the alert report actionable enough to continue with a paid production plan.
