# TODO

* intro
* problem formulation
* policy learning as operator learning
* numerical experiments
    * tracking problem:
        * heat 1d
        * fisher kpp
        * heat 2d
        * ablation studies
            * centralized vs decentralized (we can say decentralized case is for problem specific problems and for the cases where the input dimension is too large)
            * zero-shot (MSE vs # agents)
            * noise-sensitivity (both noise in sensor and in actuators)
            * kinetic vs fixed actuators
            * others
    * stabilization problems
        * ks1d
        * ks2d
        * turbulence 2d
        * ablations
            * zero-shot
            * kinetic vs fixed
            * noise sensitivity
            * others
    * pattern formation
        * ad/ns2d
        * ablations



todo:
- aggiungere noise ad actuators output e actuators input
    - vedere che cosa non funzia mo' se abbiamo fatto un casino
    
- aggiungere possibilità di avere sia fixed actuators sia kinetic
- aggiungere sensor dimension come parametro (potenzialmente)

- domande per domani con jan
    - noise sia per actuators sia per observation?
    - shall we do all the ablations for all the pdes or maybe just for one pde for each task
    - high density treated as a "noisy" variation: smoother policy results in higher less osciallatory behaviours when the density increase
    - we have oscillations as seen in picture


- meeting notes:
    - dibakar's presentation
    - my tests
    - what we have, where we are
    - ablations
        - fkpp
            - sensor dimension
            - noise and zero-shot interplay
        - 