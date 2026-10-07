# Completed corpus integrity diagnosis and isolated replacement

Audit 514055 completed all 6081 files with one decode failure and one blob-name
hash discrepancy, both at common_crawl-science_math_and_technology-0015/shard_00001148.jsonl.zst.
6080 streams decode completely and match historical download hashes.
Historical manifest checks alone did not establish upstream file authenticity.

Registered isolated repair job 514077 authenticated pinned upstream LFS SHA256
and size, downloaded to a separate cache, matched complete Python-zstandard and
Arrow decoded SHA256 0b55f33fdd0aeb693ab28b58cd683a5f792ab33668d3e0e2a620954a6d42edeb, and
validated all 47233 documents. Old Arrow stream also fails after 100663296 bytes.
This confirms a corrupt local cached artifact. Original bytes and manifests were
preserved. Replacement result SHA256 a9050e57c6ae06615795420232a13bb635ac3c71630ac28488db8e2d7160bc4d; full metadata in research-state.yaml.

Audit and repair costs are reconciled at 0.0286111111 and 0.0030555556 H200-hours.
A verified isolated replacement is not a complete production split. New005
repair import must verify lineage, current hashes and unchanged canonical order
before rerunning split, eval archival and filtering. No production cleanup yet.
