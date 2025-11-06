### Documentation of Saliency Evaluation Metrics

This document details the quantitative metrics used to evaluate the performance of Salient Object Detection (SOD) and Dichotomous Image Segmentation (DIS) models. These metrics compare the model's continuous prediction map (`prediction`) with a binary ground truth map (`ground_truth`).

---

### 1. Mean Absolute Error (MAE)

*   **What it Measures**: MAE computes the average pixel-wise absolute difference between the normalized prediction mask and the ground truth mask. It provides a direct, straightforward measure of the overall error.

*   **Formula**:
    $$ \text{MAE} = \frac{1}{W \times H} \sum_{x=1}^{W} \sum_{y=1}^{H} | P(x,y) - G(x,y) | $$
    where `P` is the prediction, `G` is the ground truth, and `W, H` are the dimensions of the mask.

*   **Interpretation**:
    *   **Lower is better**. A score of **0** indicates a perfect pixel-for-pixel match. A higher score indicates a greater average error across the entire image.

*   **Relevance and Justification**: MAE is a fundamental metric that penalizes all errors equally, regardless of their location (e.g., at the center of an object vs. its boundary). While its simplicity is a strength, this lack of perceptual awareness is also its primary limitation. It serves as an excellent baseline for measuring raw prediction accuracy. Its use in saliency evaluation was solidified in early benchmark papers that sought a simple way to quantify prediction error.

*   **Bibliographic Citation**:
    > Perazzi, F., Krähenbühl, P., Pritch, Y., & Hornung, A. (2012). "Saliency filters: Contrast based filtering for salient region detection." In *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*, pp. 733-740.

---

### 2. Maximum F-measure (Max F&#x03B2;)

*   **What it Measures**: The F-measure is the harmonic mean of Precision and Recall, providing a score that balances the two. In saliency evaluation, the continuous prediction map is converted to a binary map using a series of thresholds (from 0 to 255). The F-measure is calculated for each threshold, and the **maximum** value obtained across all thresholds is reported. This adaptive thresholding makes the metric robust to models that produce well-shaped but poorly-calibrated (e.g., generally too dark or bright) saliency maps.

*   **Interpretation**:
    *   **Range**:.
    *   **Higher is better**. A score of **1** represents a perfect balance of precision and recall at the optimal threshold.

*   **Relevance and Justification**: Unlike the pixel-level MAE, the F-measure is region-based. It evaluates how well the *shape* of the predicted salient region aligns with the ground truth. By finding the optimal threshold for a given prediction, it fairly assesses the quality of the saliency map's structure, forgiving issues with overall intensity. The standard beta-squared value ($&#x03B2;^2$) is set to **0.3** to weigh precision more heavily than recall, as proposed by the authors of the foundational paper below.

*   **Bibliographic Citation**:
    > Achanta, R., Hemami, S., Estrada, F., & Süsstrunk, S. (2009). "Frequency-tuned salient region detection." In *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*, pp. 1597-1604.

---

### 3. S-measure (Structure-measure, S&#x03B1;)

*   **What it Measures**: The S-measure is a more advanced metric designed to overcome the limitations of pixel- and region-based metrics by evaluating structural similarity. It combines two components: an **object-aware structural similarity** (evaluating the similarity between the predicted foreground and the ground truth object) and a **region-aware structural similarity** (evaluating the similarity between sub-regions of the maps).

*   **Interpretation**:
    *   **Range**:.
    *   **Higher is better**. A score of **1** indicates a perfect structural match, meaning the prediction captures the true form of the salient object much better than a simple "blob" would.

*   **Relevance and Justification**: The S-measure was introduced because previous metrics could give high scores to predictions that correctly located a salient object but failed to capture its detailed structure. This metric is less sensitive to minor pixel errors and more attuned to the geometric and structural correctness of the prediction, making it highly aligned with human visual perception.

*   **Bibliographic Citation**:
    > Fan, D. P., Cheng, M. M., Liu, Y., Li, T., & Borji, A. (2017). "Structure-measure: A new way to evaluate saliency maps." In *Proceedings of the IEEE international conference on computer vision (ICCV)*, pp. 4548-4557.

---

### 4. E-measure (Enhanced-alignment measure, E&#x03BE;)

*   **What it Measures**: The E-measure simultaneously captures both image-level statistics (the global mean) and local pixel-matching information. It computes an "enhanced alignment matrix" that quantifies the relationship between the prediction and ground truth, considering both their values and their statistical context.

*   **Interpretation**:
    *   **Range**:.
    *   **Higher is better**. A high score indicates strong alignment in both the fine-grained pixel details and the overall statistical distribution of saliency values.

*   **Relevance and Justification**: The E-measure provides a more holistic evaluation than F-measure or MAE. It was designed to be robust and to capture a wider range of similarities and differences that might be missed by other metrics. Its ability to combine local and global aspects makes it a powerful tool for comprehensively assessing a model's performance.

*   **Bibliographic Citation**:
    > Fan, D. P., Gong, C., Cao, Y., Ren, B., Cheng, M. M., & Borji, A. (2018). "Enhanced-alignment measure for binary foreground map evaluation." In *Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 698-704.

---

### 5. Weighted F-measure (F&#x03B2;w)

*   **What it Measures**: This is a modification of the F-measure that introduces a weighting scheme to address a key flaw: not all pixels are equally important. It assigns higher importance to pixels near the center of a salient region and less importance to those near the boundary. Errors made on perceptually important pixels are therefore penalized more heavily.

*   **Interpretation**:
    *   **Range**:.
    *   **Higher is better**. A high score indicates high precision and recall, especially in the most critical areas of the salient object.

*   **Relevance and Justification**: The motivation behind the Weighted F-measure is to create a metric that better correlates with human perception of error. A small error in the middle of an object is often more jarring than an error along its noisy edge. By using a weighting map (often based on centrality or distance from the boundary), this metric provides a more perceptually accurate evaluation of a saliency map's quality.

*   **Bibliographic Citation**:
    > Margolin, R., Zelnik-Manor, L., & Tal, A. (2014). "How to evaluate foreground maps?" In *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*, pp. 248-255.

### Summary Table

| Metric | What it Measures | Range | Goal | Key Strength |
| :--- | :--- | :--- | :--- | :--- |
| **MAE** | Average pixel-wise error | | **Minimize** | Simple, direct error measurement. |
| **Max F-measure** | Optimal balance of precision and recall | | **Maximize** | Robust to intensity calibration issues; region-based. |
| **S-measure** | Structural and object-level similarity | | **Maximize** | Evaluates the geometric correctness of the prediction. |
| **E-measure** | Global and local pixel alignment | | **Maximize** | Provides a holistic score combining local and global stats. |
| **Weighted F-measure** | Perceptually weighted precision and recall | | **Maximize** | Aligns better with human perception by weighting errors. |