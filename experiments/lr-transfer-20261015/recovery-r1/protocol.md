# 單段 WSD 的1–6B token LR transfer

使用者要求依 arXiv:2608.20061v1 設計並執行實驗、推導1/2/3/4/5/6B token的LR，並選擇單段WSD。
本研究只做固定模型的token-horizon transfer，不宣稱μP/model-width transfer。

## 固定條件與LR定義

- 架構Llama3.2-1B，實際1,235,814,400 parameters，8192 context、bf16、FA3、activation checkpoint。
- residual-half初始化、seed42、raw preparation dolma-raw-baseline-20261008；每個run從隨機權重開始，不載入training checkpoint。
- 8H200/8CPU、microbatch1、accumulation2、global batch16、131072 nonpadding inputs/update。
- 固定Muon/AdamW groups、WD、momentum/betas、clipping及原始canonical seed42 split。使用相同1000-document held-out subset，1479302 prediction tokens。
- 單段WSD沿用已驗證control的wsd_hold_zero形狀：warmup170updates、40%第一段decay、10%平台（峰值20%）、10%最後decay至零。不是resplit。
- LR是optimizer設定的峰值。Muon的native dimension adjustment保持original；AdamW/Muon固定比值0.03。
- 本研究尋找這條固定比值路徑上的共同最佳倍率，不聲稱兩個optimizer LR的二維全域最佳值。

## 資源與順序

| 階段 | candidates | 每run allocation上限 | token endpoint | H200-hours上限 |
|---|---:|---:|---:|---:|
| full-model GPU正確性驗證 | 1 | 10min | 8updates | 1.333333 |
| LR數值穩定性screen | 4 | 15min | 約120M，完整batch | 8 |
| stable/EMA horizon measurement | 4 | 90min | 1,000,079,360 | 48 |
| EMA對單段WSD的短程驗證 | 3 | 45min | 400,031,744 | 18 |

合計最多75.333333 H200-hours，納入原init-recipe-20261014的256H pool。
開始前已結算176.186667H，最壞合計251.52H；不重設歷史成本。所有新run是有界的transfer pilots，沒有額外formal run。
同階段平行；下一階段必須完成Slurm結算與endpoint驗證。任何失敗/未完成候選停止自動晉級；不自動retry/resume。
不自動啟動1–6B正式訓練，不清理artifact、不上傳HF。大檔及checkpoint均在/work/asciibase64/。

## LR網格與EMA

固定anchor：Muon0.005974384411214128、AdamW0.00017923153233642382。
共同倍率0.35/0.7/1.4/2.8；四個run都需成功，不能用screen loss剔除而留下偏倚網格。
GPU驗證先檢查token-spaced EMA recurrence、measurement swap還原、RNG與EMA state，再跑full-model短訓練。

Stable proxy warmup170updates後保持峰值，不做提前cooldown。EMA beta0.6、每32updates（4,194,304 tokens）更新，從早期開始；每384updates（50,331,648 tokens）評估raw及EMA。
這是相對proxy horizon縮放的EMA cadence，非直接沿用論文的2B間隔。EMA只供proxy測量；真正WSD validation完全關閉EMA。
因此不把既有「只在最後20%啟動EMA」的sweep混入本研究。

## 推導與預先指定驗證

1. 在每個相同token endpoint，以EMA held-out mean NLL（非training loss、非平均batch PPL）擬合：
   L(eta,B) = a_B [log eta]^2 + b_B log eta + c_B。
2. eta*(B) = exp(-b_B/(2a_B))。要求a_B>0、measured最低點與vertex均在LR網格內、quadratic R²>=0.9。
3. 只用約200–650M的有效點，擬合log eta*(B) = beta log(B/1B) + gamma。
4. 保留約750M–1B作retrospective held-out token validation。要求至少4點、擬合R²>=0.8、median relative LR error<=15%、最大<=25%。
   這些門檻是本研究預先指定的工程判準，不是論文聲稱的通用界限。相鄰checkpoint相關，不當作獨立seed。
5. 通過後用全部有效點refit，產生C=exp(gamma)：Muon_LR(B)=C*(B/1B)^beta；AdamW_LR(B)=0.03*Muon_LR(B)。
6. 400M單段WSD使用預測LR的1/sqrt(2)、1、sqrt(2)倍，三個新run皆fresh、EMA OFF、完整cooldown。
   要求實測最低值在中間，quadratic optimum與proxy預測相差<=25%。失敗則將1–6B表標為未驗證估計，不偽造最佳值。
7. 1B附近在proxy測量範圍；2–6B仍是外推，並未逐一實測。報告early/all/late fit視窗敏感度範圍，不把它叫95%信賴區間。

新增/改過的數值邏輯僅存在隔離source分支。已完成的2B source、checkpoint、結果及controller不修改。

## 輸出

controller-state.json及results.json提供進度、每個job/W&B與實際/保留費用。
proxy-fit.json保存每個horizon的四個LR/NLL、曲率、vertex、R²、held-out預測誤差。
predictions.json及lr-table.md列出1/2/3/4/5/6B的Muon與AdamW peak LR、fit-window範圍與驗證狀態。
report.md整合所有結果。GPU階段由sealed slurm/lr_transfer_controller.py的唯一owner每分鐘執行；cron保留其他排程。
