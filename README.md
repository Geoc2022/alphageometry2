# AG2 + Grobner Basis

Here AlphaGeometry2 is used alongside a Gröbner basis solver to solve some problems, building on the basic Gaussian elimination solver used in the original AlphaGeometry2. Essentially this allows AR to reason on polynomial based problems instead of just linear problems. For example, this version of AlphaGeometry2 can solve problems that require the Pythagorean theorem. This also could be extended to problems that require power of the point or area of a triangle (using Heron's formula).

Now the system can derive the [Parallelogram law](https://en.wikipedia.org/wiki/Parallelogram_law), which is a consequence of the Pythagorean theorem. The parallelogram law states that for any parallelogram, the sum of the squares of the lengths of the four sides is equal to the sum of the squares of the lengths of the two diagonals.

![para_law](https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Parallelogram_law_squares.svg/500px-Parallelogram_law_squares.svg.png)

In other words, the area of the blue squares is equal to the area of the red squares.

The system can also derive [Stewart's theorem](https://en.wikipedia.org/wiki/Stewart%27s_theorem), which is a generalization of the Pythagorean theorem. Stewart's theorem states that for any triangle, and a cevian (a line segment from a vertex to a point on the opposite side), the following relationship holds:
$$b^2 m + c^2 n = a(d^2 + mn)$$

![stewart](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Stewarts_theorem.svg/3840px-Stewarts_theorem.svg.png)

Tests for this extension are at [test_poly.py](https://github.com/Geoc2022/alphageometry2/blob/grobner/test_poly.py)

## Getting Started

This repo uses [`uv`](https://docs.astral.sh/uv/) to manage the environment. After cloning the repo, run the following to get set up.

```bash
uv sync
source .venv/bin/activate
```

Now you can run the test file with:

```bash
python test_poly.py
```

## References

https://en.wikipedia.org/wiki/Gr%C3%B6bner_basis

https://www.idealsvarietiesalgorithms.org/

https://pi.math.cornell.edu/~dmehrle/notes/old/alggeo/15BuchbergersAlgorithm.pdf

https://static1.squarespace.com/static/64f4f4bbed9a5e630f983d0f/t/64f55b151e67627eb1d90227/1693801238043/f4.pdf


# Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2

We present AlphaGeometry2, a significantly improved version of AlphaGeometry
introduced in [Trinh et al. (2024)](https://www.nature.com/articles/s41586-023-06747-5),
which has now surpassed an average gold medalist in solving Olympiad geometry
problems.

**Update (Jan 2026):** The AG2 paper is published in [JMLR](https://www.jmlr.org/papers/v26/25-1654.html).

This repository contains code necessary to reproduce solving some olympiad
geometry problems (for example, IMO 2005 P1, IMO 2008 P6, IMO 2013 P3 and
others).
This is done by running the symbolic part of AlphaGeometry2 system called DDAR.
DDAR runs on problems formalized into the AlphaGeometry2 language and
supplemented by the point coordinates. Some problems can be solved by DDAR
alone and others require auxiliary points provided by a language model.

## Installation

Installation is done in a virtual environment:

```
$ sudo apt install python3
$ sudo apt install python3-pip
$ sudo apt install python3-venv
$ python3 -m venv ag2
$ source ag2/bin/activate
$ pip install numpy
```

## Usage

Command below runs the prover on some IMO problems solved by AlphaGeometry2:

```
$ python -m test
```

You will see the following output:

```
We run the logical core DDAR on some easier IMO problems that can be solved by DDAR alone.

Problem: 2000_p1
..... Proven :-)

Problem: 2002_p2a
.... Proven :-)

Problem: 2002_p2b
.... Proven :-)

Problem: 2003_p4
... Proven :-)

Problem: 2004_p5
... Proven :-)

Problem: 2005_p5
...... Proven :-)

Problem: 2007_p4
.... Proven :-)

Problem: 2010_p4
.... Proven :-)

Problem: 2012_p1
...... Proven :-)

Problem: 2013_p4
..... Proven :-)

Problem: 2014_p4
.... Proven :-)

Problem: 2015_p4
... Proven :-)

Problem: 2016_p1
..... Proven :-)

Problem: 2017_p4
..... Proven :-)

Problem: 2022_p4
..... Proven :-)


We run the logical core DDAR on some challenging IMO problems, with manually provided
auxiliary points. This is not a full AlphaGeometry system, only a test of the logical core.

Problem: 2001_p5a
......... Proven :-)

Problem: 2003_p4b
..... Proven :-)

Problem: 2005_p1
...... Proven :-)

Problem: 2008_p1b
........ Proven :-)

Problem: 2008_p6
...... Proven :-)

Problem: 2009_p4a
..... Proven :-)

Problem: 2009_p4b
...... Proven :-)

Problem: 2011_p6
...... Proven :-)

Problem: 2013_p3
....... Proven :-)

Problem: 2019_p2
.... Proven :-)

Problem: 2021_p3
......... Proven :-)
```

## Citing this work

Please cite this work as:

```
@article{chervonyi2025gold,
  title={Gold-medalist performance in solving olympiad geometry with alphageometry2},
  author={Chervonyi, Yuri and Trinh, Trieu H and Ol{\v{s}}{\'a}k, Miroslav and Yang, Xiaomeng and Nguyen, Hoang H and Menegali, Marcelo and Jung, Junehyuk and Kim, Junsu and Verma, Vikas and Le, Quoc V and others},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={241},
  pages={1--39},
  year={2025}
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
