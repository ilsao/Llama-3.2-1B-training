2026-10-19：固定教師的 producer–consumer pipeline 可行性分析。這是設計與尺寸估算，沒有啟動新 allocation 或改動正在訓練的配置。

兩個常駐 Slurm 作業分別執行教師 producer 與學生 consumer；教師一次載入，依 canonical batch manifest 持續計算。初始預填1–2批後學生開始，佇列滿時教師等待，學生完成使用後 ACK 才釋放／刪除該 slot。不要每個batch重新提交 sbatch 或重新載入教師。Slurm不保證新作業立即取得資源，也不會自動搬移一般檔案；兩個allocation的啟動時間及等待仍需計入成本。見 [Slurm sbatch](https://slurm.schedmd.com/sbatch.html)。

固定 teacher eval/no_grad、固定輸入與 document masks 讓教師可超前學生更新，無 student-weight staleness。所有slot帶run ID、batch ID、canonical資料位置、token/mask checksum、teacher revision與dtype；寫入獨立partial檔、完成後原子publish READY，學生校驗且完成使用後ACK。容量上限必須包含writing、ready及reading三種狀態，不能只數ready檔。等待要有deadline、heartbeat與故障訊號；一方失敗時另一方停止等待，保留token及allocation帳目。

使用128256 vocabulary、8192 context的8191預測位置、BF16，每sequence完整logits為1.956792GiB。跨8學生ranks的一輪microbatch為15.654339GiB；global BS16為31.308678GiB、BS64為125.234711GiB。兩個global-batch slots共62.617355／250.469421GiB，尚未計入傳輸或其他暫存。用固定microbatch／chunk容量的佇列可避免BS切換使暫存擴大；跨rank梯度累積窗口仍依原training manifest。

在約一秒一個BS16更新的目標速度下，磁碟方案每秒要寫33.617GB、讀33.617GB。提前刪除限制存放量，無法消除讀寫量。若共享儲存的讀寫共用瓶頸，合計約67.235GB/s；若有RAM/page cache，後端磁碟流量可能較少，但GPU↔host、網路與記憶體搬運仍存在。nano4實際路徑吞吐尚未量測，不能宣稱可完全隱藏。

理想充分重疊的穩態耗時下限為max(教師產生、傳輸、學生訓練)，各項須以同一批tokens衡量。預填只能吸收啟動與短期抖動；長期producer較慢時任何固定容量佇列終會耗盡。填管、排程、eval期間背壓和結尾排空仍有時間成本。分開GPU可能減少目前同卡stream競爭，但若總GPU固定8張，教師專用卡會減少學生卡數。假設改為8學生＋4教師，同時間費用變1.5倍，至少1.5倍速度才能打平GPU-hours；此額外資源不是本設計已啟動的配置。

優先測試的傳輸格式：教師最後一層、final norm之後的hidden states [positions,3072]，學生端保留凍結教師lm_head [128256,3072] 並分塊重建完整BF16 logits，仍算原 .75 CE + .25 KL(teacher‖student)，T=1。由於目前teacher路徑本來就是hidden @ teacher_head.T，這在數學上保留完整分布，沒有top-k截斷；必須維持輸出層權重、final norm位置、shift、dtype及kernel/chunk規格並驗證數值誤差。

此格式每BS16約0.749908GiB、BS64約2.999634GiB，兩slot約1.499817／5.999268GiB；傳輸量縮小41.75倍。每個需要重建logits的學生GPU另存約0.733887GiB凍結教師輸出層，且承擔其projection計算。約一秒一BS16時單向資料約0.805208GB/s。它把主要教師backbone移到producer，沒有消除教師output projection或學生full-vocabulary loss成本。

傳輸可選共享檔案、共享host RAM（僅實際同節點時）或GPU tensor point-to-point；PyTorch2.9 NCCL提供CUDA tensor的send/recv及非同步P2P介面，仍需正確stream lifetime與收發順序，實際跨節點網路與排程可用性未驗證。見 [PyTorch2.9 distributed](https://docs.pytorch.org/docs/2.9/distributed.html)。同allocation分GPU角色可避免兩個獨立job的起跑時間差；是否更快仍受資源分配與實際傳輸限制。

下一個合理驗證是短Slurm benchmark：相同canonical窗口與global batch，比較目前online pipeline、完整logits bounded queue、hidden-state bounded queue。先核對loss及學生梯度，再量producer/consumer tokens/s、傳輸耗時、等待比例、peak memory及整體allocated GPU-hours/token。先測GPU tensor或host memory路徑，只有確有必要才加入大量共享磁碟流量；所有重型量測都在compute allocation執行。未量測前不保證wall time或GPU-hours改善。
