# SpatialSENet

A Dual-Microphone Two-Stage Speech Enhancement Framework for Audio Zooming

---

### Framework Overview

**SpatialSENet** is a two-stage cascaded audio-visual zooming framework tailored for dual-microphone smartphones. It addresses the challenge of limited spatial resolution and constrained computational resources by decoupling spatial filtering and spectral refinement. The system achieves consistent improvements under both anechoic and reverberant conditions, providing a perceptually smooth audio zoom effect that aligns what you "see" with what you "hear".

### Key Features

* **Two-Stage Cascaded Design**: Decouples spatial filtering from spectral refinement to ensure stable training and robust spatial selectivity.


* **Directionally Guided SpatialNet (DG-SpatialNet)**: Explicitly injects field-of-view (FoV)-driven directional priors into the network by comparing observed inter-microphone phase differences (IPDs) with theoretical IPDs.


* **Knowledge-Distilled Refinement (KD-MP-SENet-Lite)**: Compresses a high-capacity MP-SENet model into an ultra-lightweight single-channel student model of just **0.22M parameters**, effectively removing residual noise and phase distortions.


* **Perceptual Audio Zooming**: Blends the enhanced target signal with the original recording under the control of a zoom factor $\lambda \in [0,1]$ to produce a smooth, continuous audio zooming transition.
