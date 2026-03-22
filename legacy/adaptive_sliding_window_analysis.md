# Core Paper Analysis: Adaptive Sliding Window (Noor et al., 2017)

## Summary

**Paper:** Adaptive Sliding Window Segmentation for Physical Activity Recognition  
**Authors:** Mohd Halim Mohd Noor et al.  
**Year:** 2017

### Key Innovation
Adaptive window sizing based on signal energy and variance characteristics.

### Algorithm
```
1. Compute window energy: E = Σ x²
2. If E changes > threshold:
   → Shrink window
3. If E stable:
   → Grow window
4. Apply to continuous stream
```

### Results
- Accuracy: 96.5% vs 91.9% (FSW)
- Improvement: +4.6%

### Limitations
- Heuristic-based thresholds
- No learning component
- Manual parameter tuning required

### Extension Opportunities
1. Learn thresholds from data
2. Use deep learning for feature extraction
3. Combine with attention mechanisms

---

*Last Updated: March 2026*
