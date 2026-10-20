# 原版對整合加速版

使用者批准實作兩組比較。A 使用 BE60P20 已封存來源，B 整合 in-place model.compile（max-autotune 含 CUDA graphs）、native Muon Newton–Schulz 核心編譯、FA3 dense mask 與 counters 調整、有界 pinned buffers／copy-stream 搬運及同步減少。原算法、精度、batch、資料順序、masks 與 scheduler 均保持。

配置的 max_steps/scheduler_steps 保留 15259，warmup 170；以 hard token target 15,728,640 停止於 120 updates，因既有 config 契約不允許 scheduler horizon 大於 max_steps。每 12 updates 評估，first 1000 reserved documents，前 20 updates 暖機、後 100 測速；所有暖機與編譯成本記帳。

Runtime 8 H200/8 CPU 最多 30 分鐘（4 H200-hours），A/B 各最多 45 分鐘（6 H200-hours），合計預留上限 16 H200-hours。只有已辨識 CUDA graph compatibility failure 可在同一 runtime allocation 切到預先配置的 no-cudagraphs；數值、mask、checkpoint、通訊錯誤停止。correctness 通過並對帳後，A 執行，B 以 afterok:A 依序執行。不存在 mode sweep、額外正式 run、模型 checkpoint continuation、自動 allocation retry、cleanup 或 HF upload。

Correctness：mean NLL relative <=0.5%、gradient L2 relative <=3%、Muon core relative <=3%；GPU 測試包含 GQA、64 與 8192、padding、structural masks、2 micro accumulation、DDP、uneven eager evaluation、checkpoint/RNG/LR replay、HF weight reload。Final A/B PPL relative <=1%；完整 benchmark 至少 5% end-to-end 加速才推薦 B。

Cache 與大 artifacts 全在 /work/asciibase64/；compiler 每 rank 1 worker，八 rank 共 8 CPU。source-b-seal 與 A 的原 source seal 在每次提交核對。
