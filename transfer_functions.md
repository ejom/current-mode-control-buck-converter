# Transfer-function derivations

This note derives the two transfer functions used in [`bode.py`](bode.py), following
Röbenack & Gerbet, *Current-Mode Control of a Distributed Buck Converter with a Lossy
Transmission Line*, Electronics 2024, 13, 3565 ([`electronics-13-03565-v2.pdf`](electronics-13-03565-v2.pdf)).
It then extends the distributed model with:

1. a discrete inductor $L_\text{in}$ in series between the switch and the line input, and
2. a discrete inductor $L_\text{in}$ in parallel with a resistor $R_\text{p}$, with that
   pair in series between the switch and the line input.

All transfer functions go from the duty ratio $D(s)$ to the current drawn from the switch
node, as used for current-mode control. Initial conditions are zero throughout.

---

## 1. Lumped (standard) buck converter — `P_lumped`, Eq. (4)

Network model (paper Fig. 2): switch voltage $E\,d(t)$, series inductor $L$ with winding
resistance $R_L$, then a shunt capacitor $C$ with insulation conductance $G_C$, parallel to
the load $R$.

**Time domain**, Eqs. (1)–(2):

$$
L\,\frac{di}{dt} + R_L\, i + v = E\,d, \qquad
C\,\frac{dv}{dt} + \left(G_C + \tfrac{1}{R}\right) v = i .
$$

**Laplace domain**, Eq. (3):

$$
\underbrace{\begin{pmatrix} sL + R_L & 1 \\ -1 & sC + G_C + \tfrac1R \end{pmatrix}}_{M(s)}
\begin{pmatrix} I(s) \\ V(s) \end{pmatrix}
= \begin{pmatrix} E \\ 0 \end{pmatrix} D(s).
$$

Write $Y(s) = sC + G_C + 1/R$ for the admittance of the output node. Then
$\det M = (sL+R_L)\,Y + 1$, and Cramer's rule gives

$$
I(s) = \frac{\det\begin{pmatrix} E & 1 \\ 0 & Y\end{pmatrix}}{\det M}\,D(s)
     = \frac{E\,Y}{(sL+R_L)\,Y + 1}\,D(s).
$$

Another way to see it: the switch sees the impedance $Z_\text{in} = sL + R_L + 1/Y$, so
$I = E D / Z_\text{in}$. This gives

$$
\boxed{P(s) = \frac{I(s)}{D(s)} = \frac{E\left(sC + G_C + \frac1R\right)}{(sL+R_L)\left(sC+G_C+\frac1R\right) + 1}}
$$

Expanding the denominator and normalising the constant term gives the paper's form (4):

$$
P(s) = \frac{E\left(sC + G_C + \frac1R\right)}
{LC\,s^2 + \left(LG_C + \frac{L}{R} + R_L C\right)s + 1 + R_L G_C + \frac{R_L}{R}} .
$$

`bode.py` uses the lumped-equivalent values from Eq. (22): $L = l L'$, $C = l C' + C_\text{ext}$,
$R_L = l R_L'$, $G_C = l G_C'$.

---

## 2. Distributed buck converter — `P_distributed`, Eq. (20)

The inductor is replaced by a lossy transmission line of length $l$ with per-length
parameters $L', R_L', C', G_C'$ (paper Fig. 3). The external capacitor $C_\text{ext}$ and
the load $R$ sit at the far end $z = l$.

### 2.1 Telegrapher's equations

Eqs. (5)–(6):

$$
L'\partial_t i + R_L' i + \partial_z v = 0, \qquad
C'\partial_t v + G_C' v + \partial_z i = 0 .
$$

Taking the Laplace transform in $t$ only, Eqs. (9)–(10), with the shorthand

$$
z_L(s) = sL' + R_L' \quad\text{(series impedance per length)},\qquad
y_C(s) = sC' + G_C' \quad\text{(shunt admittance per length)},
$$

gives an ODE in the spatial variable $z$, Eq. (11):

$$
\frac{\partial}{\partial z}\underbrace{\begin{pmatrix} I(z,s) \\ V(z,s)\end{pmatrix}}_{X(z,s)}
= \underbrace{\begin{pmatrix} 0 & -y_C \\ -z_L & 0 \end{pmatrix}}_{A(s)} X(z,s).
$$

### 2.2 Propagation constant and matrix exponential

The characteristic polynomial is $\det(\gamma I - A) = \gamma^2 - z_L y_C$, so, as in Eq. (12),

$$
\gamma(s) = \sqrt{z_L\,y_C} = \sqrt{(sL'+R_L')(sC'+G_C')} .
$$

The solution is $X(z,s) = e^{A z} X(0,s)$, Eq. (13). Because $A^2 = z_L y_C\, I = \gamma^2 I$,
the even powers of $A$ in the exponential series are scalar multiples of $I$ and the odd
powers are scalar multiples of $A$. So, as in Eq. (15),

$$
e^{Az} = \sum_k \frac{(Az)^k}{k!}
= \underbrace{\sum_{k\ \text{even}} \frac{(\gamma z)^k}{k!}}_{\cosh\gamma z} I
+ \frac1\gamma\underbrace{\sum_{k\ \text{odd}} \frac{(\gamma z)^k}{k!}}_{\sinh\gamma z} A .
$$

### 2.3 Map the load end back to the input

The far-end boundary condition (8), $C_\text{ext}\dot v(l) + v(l)/R = i(l)$, becomes Eq. (16):

$$
V(l,s) = Z(s)\, I(l,s), \qquad Z(s) = R \parallel \frac{1}{sC_\text{ext}} = \frac{R}{1 + sRC_\text{ext}} .
$$

To express the input end in terms of the output end, invert the propagation:
$X(0,s) = e^{-Al} X(l,s)$, Eq. (17). Using (15) with $z = -l$:

$$
M(s) := e^{-Al} = \cosh(\gamma l)\, I - \frac{\sinh(\gamma l)}{\gamma} A
= \begin{pmatrix}
\cosh\gamma l & \dfrac{y_C}{\gamma}\sinh\gamma l \\[2mm]
\dfrac{z_L}{\gamma}\sinh\gamma l & \cosh\gamma l
\end{pmatrix}.
$$

(This is the familiar ABCD/chain matrix of a line with $Z_0 = \sqrt{z_L/y_C} = z_L/\gamma = \gamma/y_C$,
written for the ordering $(I, V)$.) Substituting $V(l) = Z\,I(l)$, as in Eqs. (18)–(19):

$$
I(0,s) = (m_{11} + m_{12}Z)\, I(l,s), \qquad
V(0,s) = (m_{21} + m_{22}Z)\, I(l,s).
$$

### 2.4 Close with the source

The near-end boundary condition (7) is $V(0,s) = E\,D(s)$. Dividing the two expressions
above cancels $I(l,s)$:

$$
P(s) = \frac{I(0,s)}{D(s)} = E\,\frac{I(0,s)}{V(0,s)}
= E\,\frac{m_{11} + m_{12} Z}{m_{21} + m_{22} Z}
= E\,\frac{\cosh\gamma l + \frac{y_C}{\gamma} Z \sinh\gamma l}
          {\frac{z_L}{\gamma}\sinh\gamma l + Z\cosh\gamma l} .
$$

Multiplying numerator and denominator by $\gamma$ gives

$$
P(s) = E\,\frac{y_C\, Z \sinh\gamma l + \gamma\cosh\gamma l}{z_L \sinh\gamma l + \gamma Z\cosh\gamma l},
$$

and multiplying both by $(1 + sRC_\text{ext})$ removes the fraction in $Z$, because
$Z(1+sRC_\text{ext}) = R$. The result is Eq. (20):

$$
\boxed{P(s) = E\,\frac{(sC'+G_C')\,R\,\sinh\gamma l + \gamma\,(1+sRC_\text{ext})\cosh\gamma l}
{(sL'+R_L')(1+sRC_\text{ext})\sinh\gamma l + \gamma\,R\cosh\gamma l}}
$$

This is `P_distributed` in `bode.py`, with `zL` $= z_L$, `yC` $= y_C$, `g` $= \gamma$.

**Square-root branch.** Under $\gamma \to -\gamma$, $\sinh$ changes sign and $\cosh$ does
not. Every term of the numerator and of the denominator then changes sign, so $P$ is even
in $\gamma$ and either branch of the square root gives the same result.

### 2.5 Useful restatement: line input impedance

For the extensions below, it is convenient to name the numerator and denominator of (20):

$$
N(s) = y_C R\sinh\gamma l + \gamma(1+sRC_\text{ext})\cosh\gamma l, \qquad
\Delta(s) = z_L (1+sRC_\text{ext})\sinh\gamma l + \gamma R\cosh\gamma l ,
$$

so $P_{20}(s) = E\,N/\Delta$. The impedance seen looking into the line (with its load
attached) is then

$$
Z_\text{line}(s) = \frac{V(0,s)}{I(0,s)} = \frac{E}{P_{20}(s)} = \frac{\Delta(s)}{N(s)} .
$$

---

## 3. Extension A — discrete inductor at the line input

### Circuit

```
 E·d ──[ L_in ]──┬──────── transmission line (L', R_L', C', G_C', l) ────────┬──┬──
                 │                                                           Cext R
 (switch node)   z = 0                                                       z = l
```

A discrete inductor $L_\text{in}$ is placed in series between the switch node and the line
input. The same current flows through the inductor and into the line, so the switch
current is $I(0,s)$.

### Derivation

Nothing changes along the line or at the far end, so Sections 2.1–2.3 still hold, and the
line input still satisfies $V(0,s) = Z_\text{line}(s)\, I(0,s)$. Only the near-end boundary
condition changes. Instead of (7), Kirchhoff's voltage law around the input loop gives

$$
L_\text{in}\frac{d}{dt} i(0,t) + v(0,t) = E\,d(t)
\quad\xrightarrow{\ \mathcal L\ }\quad
s L_\text{in} I(0,s) + V(0,s) = E\,D(s).
$$

Substituting $V(0) = Z_\text{line} I(0)$:

$$
P_A(s) = \frac{I(0,s)}{D(s)} = \frac{E}{sL_\text{in} + Z_\text{line}(s)}
= \frac{E}{sL_\text{in} + \Delta/N} = \frac{E\,N}{sL_\text{in} N + \Delta} .
$$

Expanding $N$ and $\Delta$ and collecting the $\sinh$ and $\cosh$ terms:

$$
\boxed{P_A(s) = E\,\frac{(sC'+G_C')\,R\sinh\gamma l + \gamma(1+sRC_\text{ext})\cosh\gamma l}
{\Big[(sL'+R_L')(1+sRC_\text{ext}) + sL_\text{in}(sC'+G_C')R\Big]\sinh\gamma l
 + \gamma\Big[R + sL_\text{in}(1+sRC_\text{ext})\Big]\cosh\gamma l}}
$$

Equivalently, as feedback around the original plant:
$P_A = \dfrac{P_{20}}{1 + sL_\text{in} P_{20}/E}$.

### Checks

* $L_\text{in} = 0$ gives back Eq. (20).
* The numerator is the same as in (20), so **the zeros do not move**. Only the poles
  (resonances) shift, because the inductor adds to the series impedance.
* The function is still even in $\gamma$ (each new term is $\sinh$ or $\gamma\cosh$), so the
  branch choice still does not matter.
* DC ($s=0$): the inductor is a short circuit, so $P_A(0) = P_{20}(0)$.
* Lumped equivalent for comparison: replace $L$ by $L + L_\text{in}$ in Eq. (4), i.e.
  $P = E\,Y / \big[(s(L + L_\text{in}) + R_L)Y + 1\big]$.

---

## 4. Extension B — inductor ∥ resistor at the line input

### Circuit

```
                ┌──[ L_in ]──┐
 E·d ───────────┤            ├───── transmission line ─────┬──┬──
                └──[ R_p  ]──┘                             Cext R
 (switch node)                z = 0                        z = l
```

$L_\text{in}$ and $R_\text{p}$ are in parallel, and the pair is in series between the switch
node and the line input. This is the classic damped input inductor: at high frequencies,
where the inductor's reactance is large, $R_\text{p}$ takes over and damps the line resonances.

### Derivation

The series element now has impedance

$$
Z_\text{in}(s) = sL_\text{in} \parallel R_\text{p} = \frac{sL_\text{in}R_\text{p}}{R_\text{p} + sL_\text{in}} .
$$

The near-end boundary condition becomes $Z_\text{in} I(0,s) + V(0,s) = E\,D(s)$. As in
Section 3,

$$
P_B(s) = \frac{E}{Z_\text{in}(s) + Z_\text{line}(s)}
= \frac{E\,N}{Z_\text{in}N + \Delta}
= \frac{E\,(R_\text{p} + sL_\text{in})\,N}{sL_\text{in}R_\text{p}\,N + (R_\text{p}+sL_\text{in})\,\Delta},
$$

where the last step multiplies through by $(R_\text{p} + sL_\text{in})$. Expanding and
collecting:

$$
\boxed{P_B(s) = E\,\frac{(R_\text{p}+sL_\text{in})\Big[(sC'+G_C')R\sinh\gamma l + \gamma(1+sRC_\text{ext})\cosh\gamma l\Big]}
{\begin{aligned}
&\Big[(R_\text{p}+sL_\text{in})(sL'+R_L')(1+sRC_\text{ext}) + sL_\text{in}R_\text{p}(sC'+G_C')R\Big]\sinh\gamma l \\
&\quad + \gamma\Big[(R_\text{p}+sL_\text{in})R + sL_\text{in}R_\text{p}(1+sRC_\text{ext})\Big]\cosh\gamma l
\end{aligned}}}
$$

### Which current?

$P_B$ is the **total** current from the switch, which is also the line input current
$I(0,s)$. If the current sensor sits in the inductor branch only, divide the current
between the two branches:

$$
I_{L_\text{in}} = \frac{R_\text{p}}{R_\text{p} + sL_\text{in}}\, I(0,s)
\quad\Rightarrow\quad
P_{B,L}(s) = \frac{E\,R_\text{p}\,N}{sL_\text{in}R_\text{p}\,N + (R_\text{p}+sL_\text{in})\,\Delta} ,
$$

i.e. the same denominator with the factor $(R_\text{p}+sL_\text{in})$ removed from the
numerator.

### Checks

* $R_\text{p} \to \infty$ (resistor removed): divide the numerator and denominator by
  $R_\text{p}$ and let $R_\text{p}\to\infty$ to get back $P_A$.
* $R_\text{p} \to 0$ or $L_\text{in} \to 0$ (series element shorted): gives back Eq. (20).
* DC: $Z_\text{in}(0) = 0$, so $P_B(0) = P_{20}(0)$.
* High frequency ($|sL_\text{in}| \gg R_\text{p}$): $Z_\text{in} \to R_\text{p}$, so the plant
  behaves like Eq. (20) with a series resistor $R_\text{p}$ in front. This adds damping
  where the line resonances are.
* The extra factor $(R_\text{p}+sL_\text{in})$ in the numerator puts a real zero at
  $s = -R_\text{p}/L_\text{in}$. The denominator does not share this factor, so the zero
  does not cancel.
* Still even in $\gamma$.
* Lumped equivalent: $P = E\,Y / \big[(Z_\text{in} + sL + R_L)\,Y + 1\big]$.

---

## 5. General pattern

Any one-port $Z_\text{in}(s)$ placed in series at the line input gives

$$
P(s) = \frac{E}{Z_\text{in}(s) + Z_\text{line}(s)} = \frac{E\,N(s)}{Z_\text{in}(s)\,N(s) + \Delta(s)} ,
\qquad Z_\text{line} = \frac{\Delta}{N} .
$$

Any one-port in place of $R\parallel C_\text{ext}$ at the far end is handled by changing
$Z(s)$ in Section 2.3.
