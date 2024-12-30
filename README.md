# 11124237 朱瓊月 ， 11123142 林韋莘
# COVID-19：使用深度學習的醫學診斷

# 介紹
正在進行的名為COVID-19 的全球大流行是由SARS-COV-2引起的，該病毒傳播迅速並發生變異，引發了幾波疫情，主要影響第三世界和發展中國家。隨著世界各國政府試圖控制傳播，受影響的人數正穩定上升。
![image](https://github.com/user-attachments/assets/904519ac-9d46-44a2-b344-2ae70cf6e84b)
本文將使用CoronaHack-Chest X 光資料集。它包含胸部X 光影像，我們必須找到受冠狀病毒影響的影像。
我們之前談到的SARS-COV-2 是主要影響呼吸系統的病毒類型，因此胸部X 光是我們可以用來識別受影響肺部的重要影像方法之一。這是一個並排比較：
![image](https://github.com/user-attachments/assets/6fbb0acd-e7fb-4cef-b886-e53bfecc05cb)
如你所見，COVID-19 肺炎如何吞噬整個肺部，並且比細菌和病毒類型的肺炎更危險。
本文，將使用深度學習和遷移學習對受Covid-19影響的肺部的X 光影像進行分類和識別。
# 導入庫和載入數據
