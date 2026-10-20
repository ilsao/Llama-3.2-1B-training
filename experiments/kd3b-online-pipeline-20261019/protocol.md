2026-10-19 使用者要求「啟動訓練」，啟動官方 pretrained Llama-3.2-3B online KD pilot；學生從隨機 residual-half seed42 初始化。這是原 methods study 的新教師實驗，舊 K1 與所有失敗、取消、驗證成本完整保留。單次 allocation 8 H200 / 8 CPU、7200 秒／16 H200-hours；整批含本次預留最多 95.5108333333 H200-hours，低於原 128 上限。沒有新增正式 full run、retry、resume 或 requeue。

教師固定 meta-llama/Llama-3.2-3B base revision 13afe5124825b4f3751f836b40dafda64c1ed062，BF16、eval、frozen、no KV cache、原 RoPE。權重 manifest SHA256 bf1dec0109c97ad0d00d5714127aca10a6046edbd0b35e0d4c1a55fb2f192507。啟動時 rank0 驗證全部權重與 tokenizer 檔案，其餘 rank 接收校驗結果。此前完整 vocabulary、tokenizer backend 與特殊 token 對齊已通過。教師 head128 使用經測試且固定版本的 Torch2.9.1 ATen varlen FlashAttention；學生保留原 FA3 head64。

學生架構為 Llama-3.2-1B，16 層、hidden2048、tied embedding/head、8192 context，FP32 參數／BF16 autocast、activation checkpointing、intradoc document-span causal mask。學生只能由 from_config 建立，禁止訓練 checkpoint 載入；教師載入與測試後重設 seed42，並驗證學生初始化 signature 與歷史 N0 相同。

目標為 .75 hard CE + .25 KL(teacher‖student)，T=1、完整 vocabulary、每塊1024位置。每張 GPU 各有一份教師與學生，2 CUDA streams／2個 BF16 logits slots。只在已預留的同一 optimizer accumulation window 內預取下一個 microbatch；學生 forward/backward 與下一批教師推論重疊。每個 update 結束必須消耗所有 ticket，evaluation／checkpoint／停止時無待用 batch。與先前跨 update 的短 benchmark 相比，這個邊界限制增加 pipeline fill；不能直接將其1.37%加速當成本次全程效能。

canonical raw Dolma 資料、seed42完整文件 shuffle、50k reserved 文件與固定1k評估保持原 identity；每個 rank 在模型初始化前讀取目標 prefix 首尾真實資料列。學生1,000,079,360 nonpadding inputs為目標／hard cap，共4775更新窗口；500,039,680 inputs 切 BS16→64；phase split833,355,776，第二 peak .1；每phase warmup170 BS16單位、40%tail、4:1:1 ramp/hold/final與.2 plateau。NorMuon NS5 matrix peak LR .008614891542046706，外層 ordinary WD .1；tied embedding/head與norms AdamW LR .0005168934925228023、betas(.9,.95)、eps1e-8、無WD；global clip .5。詳細配置見 config.json，未調整 recipe。

學生與教師各有獨立 persistent input journal，兩者均先預留再執行、即使失敗也保留 charged inputs。教師窗口含未完成預取的保守預留；成功窗口才 confirm。正常訓練每個教師 input 對應一個學生 input，最多合計2,000,158,720模型inputs。固定1k教師評估另存收據，不併入訓練inputs。沒有 offline logits cache；GPU暫存重複使用。

先前 allocation573794 已驗證實際3B教師＋1B學生完整8rank梯度一致性、不同hidden維度、BF16 loss/backward、文件隔離、padding、FIFO與slot重用。新allocation再次執行小型GPU正確性檢查及真實loader預檢，再量測教師固定1k PPL，最後建立隨機學生。首次真實更新沿用已驗證的 NorMuon/AdamW 參數覆蓋及權重更新。

W&B project lab5-training-llama，新 identity。train/loss 為整個logging interval的token加權hard CE；train/distillation_objective、train/teacher_kl、train/hard_ce 為目前更新的值。required grad_norm為同步DDP梯度clip前norm；tokens_per_second以CUDA同步最慢rank計時，包含教師、學生、transfer與optimizer，不含evaluation或journal I/O。另記完整window wall time、GPU memory、teacher inputs、各group LR與decay。

固定1k每400更新、原token knots、wall-time間隔及停止時評估；PPL取 exp(sum NLL/sum valid labels)。只有完成1B目標才執行同allocation額外9k確認，保留逐文件收據。不以提前停止結果填入1B終點比較。初始化、teacher eval、訓練、評估、checkpoint、HF格式本地export均計入2小時；保留900秒額外評估預算及300秒停止margin。依先前速度估算，成熟3B教師可能無法在此allocation完成1B，屆時保留exit75和實際進度，不延長、不自動續跑。

source/ 是新隔離快照；歷史封存程式不變。slurm/kd3b_online_pipeline.py 先驗證 seal、記錄提交intent、shared-ledger reserve，再sbatch及bind。遇不確定提交保留reservation並找回原job，不建立第二allocation。完成後使用同工具 reconcile 記帳。禁止artifact cleanup、HF upload。
