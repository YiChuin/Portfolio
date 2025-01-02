本頁面為邱奕銓在學期間累積的比賽專案作品與其他專案作品，以下為各個作品說明


# 專案作品集
此目錄存放各項專案作品的簡報與報告書，其中包含了以下內容:

1. 交通部113年國道智慧交通管理創意競賽時間預測分析與探討
   
   **研究目的 :** 為了避免因道路壅塞增加事故發生機率或二次交通事故，欲解決國道雍塞問題並及時發布預計事故排除時間等相關訊息，提醒用路人改道

   **研究方法 :** 建立模型預測事故排除時間，並結合物件辨識與追蹤模型、語音辨識技術與大型語言模型來舒緩國道交通量問題
   
    - 利用交通部事故資料集訓練LightGBM來預測事故排除時間，並且預測平均誤差僅10分鐘
    - 提出利用LLM、RAG與OpenAI的Whisper API，串聯交通部資料庫，建構語音系統，改良交通部1968APP，使駕駛人能直接透過語音詢問路況或交通法規

2. 玉山人工智慧公開挑戰賽－RAG與LLM在金融問答的應用

   **研究目的 :** 隨著大型語言模型的問世，金融業龐大且複雜的資料已經不再是語料檢索無法高度泛化的障礙，而是逐漸被解決的問題。因此我們嘗試建立高準確度的檢索方式，使公司端能更好的去尋找相關文檔
   
   **研究方法 :** 設計系統機制以提高檢索結果的準確性，包括從提供的語料中找出完整回答問題的正確資料
   
   - 根據使用者的問題檢索圖像與文字等多模態的財務相關資料，使其後續能用於提升生成式AI的回答品質
   - 將⽟⼭銀⾏保險產品之保單條款與部分台股上市公司財務報資料切Chunk，並進行Embedding儲存成向量庫
   - 結合向量庫檢索和Rerank Model使準確度從只用向量庫檢索的78%提升至89.49%


3. 自動作文評分系統

   **研究目的 :** 文章寫作在教育上一直都是評估學生學習和表現的重要方法，然而對於教育工作者而言，批改文章這件事相當的耗時且費力。因此我們嘗試建立一個自動寫作評估（AWE）系統來對文章進行評分，用以減輕教育工作者的工作負擔。
   **研究方法 :** 設計機制以提高檢索結果的準確性，包括從提供的語料中找出完整回答問題的正確資料
   - 利用作文文本預測該作文的級分，並有著80.87%的準確度
   - 觀察高低分文章的常用詞彙、詞頻分析、詞性分佈、長度分布和主題等，並用於後續的資料處理
   - 對作文文本和對應級分資料集利用NLTK做資料增強
   - 微調DeBERTa-V3 small模型，並且輸出混淆矩陣觀察預測狀況
   


4. 利用深度學習進行的竊電行為檢測

   **研究目的 :** 全球每年因電力盜竊造成的經濟損失高達近千億美元。電力盜竊問題不僅對經濟造成巨大損失，還可能影響電力系統的穩定性和安全性。而現行偵測竊電的研究大致分為四個方向，賽局理論、電網分析、硬體分析、機器學習，由於機器學習以外                  都有相當程度的限制，因此我們主要探討機器學習相關的方法
   **研究方法 :** 使用機器學習演算法，僅從用戶的每日用電量去評估是否有竊電的行為。將資料以每周為一列排序成類似圖片的樣子，透過捕捉每周用電量的週期性去進行判斷是否有異常
     - 使用中國國家電網公司 (SGCC) 釋出的竊電資料集，從用戶的每日用電量去評估是否有竊電的行為，以及評估預測的可信度
     - 利用用電量週期性的規則去做缺失值填補(原資料集有高達25%以上的缺失值)
     - 使用寬深CNN的架構，利用用電的週期性去判斷是否為異常值，AUC達80.2%


