# 配置入口

目前選定的配置是 [6B R5 KD](../recipes/sota6b/config.json)，其執行入口是
[recipe implementation](../experiments/kd6b-r5-20261020/runner.py)。
配置保留在原實驗封存位置，recipe 以 symlink 引用，避免版本漂移。

本目錄原有 JSON 是早期 dataset、baseline 與 sweep 配置；檔名與內容維持相容。
各日期實驗的 resolved configuration、authorization、digest 位於對應 experiments/ 目錄。
根目錄 train.py 是早期 baseline，不能只換成 R5 config 就重現 R5 trainer。
