DyF-ZTL: Federated Dynamic Fuzzy Learning for Privacy-Preserving Zero Trust Endpoint SecurityThis repository contains the official implementation of the paper "DyF-ZTL: Federated Dynamic Fuzzy Learning for Privacy-Preserving Zero Trust Endpoint Security".

Abstract: The rapid proliferation of Windows-based endpoints in enterprise networks has necessitated security paradigms beyond traditional perimeter defenses. 


To address the conflict between Zero Trust Architecture (ZTA) verification requirements and privacy regulations (e.g., GDPR), this paper proposes DyF-ZTL.DyF-ZTL is a novel framework that integrates:Client-Side: A Dynamic Neuro-Fuzzy layer with learnable membership functions that adaptively capture non-stationary user behaviors via gradient descent1.Server-Side: An Adaptive Trust Engine featuring a "Draconian Decay" mechanism ($\lambda=0.2$) that enforces the immediate isolation of compromised nodes.

Dataset : https://research.unsw.edu.au/projects/toniot-datasets


Installation
Prerequisites
Python 3.8+
PyTorch
Scikit-learn
NumPy, Pandas
