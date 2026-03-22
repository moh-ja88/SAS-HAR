# Core Paper Analysis: Statistical Similarity Segmentation (Baraka et al., 2023)

## Summary

**Paper:** Similarity Segmentation Approach for Sensor-Based Activity Recognition  
**Authors:** AbdulRahman Baraka, Mohd Halim Mohd Noor  
**Year:** 2023  
**Venue:** IEEE Sensors Journal

### Key Innovation
Treat segmentation as binary classification of window similarity.

### Algorithm
```
1. Extract statistical features from windows
2. Compute similarity between adjacent windows
3. If similarity < threshold:
   → Mark as boundary
4. Segment stream at boundaries
```

### Features
- Mean, variance, energy
- Skewness, kurtosis
- Correlation coefficients

### Similarity Metrics
- Euclidean distance
- Cosine similarity
- DTW (Dynamic Time Warping)

### Results
- Boundary F1: 82-86%
- Accuracy: 91-94%

### Limitations
- Hand-crafted features
- Threshold sensitivity
- Local context only (adjacent windows)

### Extension to Deep Similarity (2024)
- Replace statistical features with CNN
- Learn similarity representations
- End-to-end training

---

*Last Updated: March 2026*
