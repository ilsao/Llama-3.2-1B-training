# WSD LR transfer：找谷底、校準proxy、驗證horizon

使用者於2026-10-15要求「你不用管時間預算，繼續設計實驗」。因此後續研究不再以原256H aggregate time budget為提交門檻；過往245.906667H及所有失敗費用完整保留。每job8H200/8CPU、每logical run6B tokens/64H、正式run數門檻、random init與禁止自動retry/resume等限制不變。

## 問題與可否證假說

既有400M WSD（EMA OFF）三點PPL：28.9105/27.4014/26.2954，最高LR最佳，尚未包住谷底。舊EMA proxy的horizon擬合R²0.9988只證明能預測EMA proxy的optimum。

- H1：WSD optimum在原網格右側。先測量真實谷底，才能計算proxy偏差。
- H2：EMA averaging window對目前131072 tokens/update不合適。固定beta0.6、每4194304tokens更新，e-folding window約8.21M tokens；它在不同horizon占比不同，不能假設與原論文32M-token batch等價。
- H3：原單段WSD有60%尾段及20%LR平台，與其他decay形狀的proxy對應不同。
- H4：proxy到WSD的LR偏移不只是常數，可能隨token horizon改變。

本輪維持1.236B dense model、residual-half、raw、context8192、global batch16、seed42及同一1000篇eval；AdamW/Muon LR比例固定0.03。品質指標為held-out mean NLL。未確認模型寬度μP transfer；未聲稱二維optimizer LR全域最優。

## A：先包住400M WSD谷底（本controller唯一自動階段）

保留舊三點，新增：

| Muon peak LR | AdamW peak LR |
|---:|---:|
|0.0117174458421985|0.000351523375265955|
|0.01657097082640935|0.0004971291247922805|
|0.023434891684397|0.00070304675053191|

每個新LR先做15min/約120M穩定性screen；三個screen均成功後，三個fresh run各做45min/400031744 tokens完整原WSD、EMA OFF。相同階段平行，每個run獨立random init、不載入舊checkpoint。
此有限stage最多24H，記帳但不受已解除的256H aggregate gate影響。沿用已通過GPU驗證的原數值payload，只有配置與提交協調器新增。

驗收：最低mean NLL必須位於完整網格內部，左右均有較差的有限值。只用最佳點及鄰點做local quadratic，列出真實觀測與網格解析度；三點的R²不當作額外證據。若最佳仍在邊界或數值不穩，輸出未包住谷底，不自動擴展無限候選。

## B：辨識EMA與cooldown造成的偏差（設計完成，等待A結果）

所有新訓練均fresh，先採用A的谷底附近至少4個共同LR；不足以包住某variant谷底時，針對該variant補一側sqrt2間距候選，最多3輪並保留失敗成本。

1. 在400M比較三種target：原60%hold-zero尾段、60%linear decay、20%linear decay。固定warmup170、模型/data/batch/optimizer參數，所有target EMA OFF。這區分尾段長度與平台形狀。
2. 另做stable proxy，保留原fixed beta0.6方案作control，測量token-relative EMA：tau(t)=rho*tokens_seen，rho={0.01,0.03,0.10}；每次snapshot的alpha(t)=exp(-delta_tokens/tau(t))。
3. 不以哪個EMA自身PPL最低選proxy；以它能否預測實際WSD的LR谷底、LR ranking及NLL regret選proxy。每個rho需記錄有效window、snapshot token cursor、數值穩定性及update-to-weight ratio。
4. 原先只保存final checkpoint與單一EMA shadow，不能從那些log重建其他EMA視窗；新proxy需要保留對應shadow/snapshot。新增heavy checks一律Slurm。

## C：用真實WSD多horizon校準（設計完成，等待A/B結果）

- Calibration horizons：0.2/0.4/0.6B，每個horizon至少4個LR完整WSD，完成cooldown。0.4B可重用A中完全相同配置觀測。
- 第一個held-out horizon：0.8B；在看到其結果前固定LR預測、候選網格與驗收門檻。
- 最終held-out horizon：1B；在看到1B WSD結果前凍結最終transfer method與1–6B表。
- 所有horizon使用相同warmup token數、資料prefix、batch、初始化、target scheduler比例；不能把2B中途checkpoint當成1B完成cooldown的觀測。
- 對每個horizon以log LR對mean NLL擬合局部谷底；跨horizon擬合log eta*_WSD=beta log(T/1B)+gamma。
- 比較三個模型：直接EMA transfer（delta=0）、固定LR偏移delta0、隨horizon變化的偏移delta0+delta1 log(T/T0)。模型選擇只用calibration horizons的leave-one-horizon-out誤差；不在最終held-out上選rho或重調參數。
- 同時保留直接WSD scaling law作為對照。EMA校準若失敗，仍可據真實WSD多horizon推導，但必須明確標示方法已不同。

## D：真正的1B SOTA對照與最後交付（設計完成）

在1B從隨機權重跑完整WSD，EMA OFF。至少比較預測LR的{0.5,1/sqrt2,1,sqrt2,2}倍，並保留A/C現有最佳recipe的匹配對照；相同run若配置完全相同可重用。這些是<=120min allocation的有界研究pilot，不自動新增正式2B/4B/6B runs。

- 必須包住1B真實谷底；預測值的NLL regret<=0.005 nats/token，且預測LR與擬合谷底相差<=15%。這些是本研究預先指定門檻，不是論文通用保證。
- predicted與最佳競爭者用seed43重做；差距很小再加入seed44。以成對document NLL檢查，而非只看train loss或單seed PPL。
- 主要評估固定前1000篇；另預先固定reserved subset接下來1000篇，只用於凍結後的確認，不參與選LR。仍不把任何eval文件放進training。
- 最終交付1/2/3/4/5/6B的Muon/AdamW peak LR、真實WSD支持範圍、LR regret、seed敏感度與外推範圍。1B以外未實測的horizon仍標外推；高R²不能替代長horizon驗證。
- 若後續需要2B以上的新增正式訓練，依實際結果另準備具體配置與正式run登記，本階段controller不會自動啟動。

## 實作與狀態

PhaseA使用獨立Git worktree與sealed source。每分鐘的wsd_bracket_controller.py是唯一提交owner；先reservation後sbatch、未知提交結果保留並recover，禁止重複提交。每個terminal allocation先sacct結算再晉級，partial不能當完成。

輸出plan.json、controller-state.json、results.json、report.md、bracket-result.json與每run W&B/receipt。原先lr-transfer-20261015的失敗判定與1–6B未驗證表保持原樣。

Reference: https://arxiv.org/html/2608.20061v1 (§2.2 and Appendix E)。
