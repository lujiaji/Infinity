from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="FoundationVision/Infinity",
    allow_patterns=["infinity_8b_weights/*"],
    local_dir="/data/jiaji_lu/Infinity/"
)
