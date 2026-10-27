# Pinned environments

- [data.txt](data.txt)：資料處理環境。
- [training.txt](training.txt)：訓練環境。
- [fa3-build.txt](fa3-build.txt)：獨立 FlashAttention3 build 環境。

檔案內容逐位元保留。根目錄 `requirements-*.txt` 是相容 symlink，舊 Slurm scripts
及 source fingerprints 可繼續使用。安裝與編譯須透過 Slurm。
6B R5 實際使用的封存依賴可由 [recipe runtime](../experiments/kd3b-hidden8-20261019/source/) 查閱；
以上通用清單不能取代各 run 的環境 provenance。
