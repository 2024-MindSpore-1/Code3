# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import datetime
import os
import math
from functools import partial
from scipy import special as sp
from scipy.optimize import brentq

import sympy as sym
import numpy as np
import mindspore
from mindspore import Tensor, nn, Parameter, ops
from mindspore import dtype as mstype
from mindspore.common import initializer as weight_init
from mindspore.common.initializer import initializer, Uniform



class Envelope(nn.Cell):
    """
    Envelope function that ensures a smooth cutoff
    """
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.exponent = exponent

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2
        self.zeroslike = ops.ZerosLike()
    def construct(self, inputs):

        # Envelope function divided by r

        env_val = 1 / inputs + self.a * inputs**(self.p - 1) + self.b * inputs**self.p + self.c * inputs**(self.p + 1)
        hh = mindspore.numpy.where(inputs < 1, env_val, self.zeroslike(inputs))
        hhh = mindspore.numpy.where(ops.IsNan()(hh), self.zeroslike(inputs), hh)
        hhhh = mindspore.numpy.where(ops.IsInf()(hhh), self.zeroslike(inputs), hhh)
        return hhhh
    
    
class BesselBasisLayer(nn.Cell):
    

    def __init__(self, num_radial, cutoff, envelope_exponent=5):

        super(BesselBasisLayer, self).__init__()
        self.num_radial = num_radial
        self.inv_cutoff = Tensor(1 / cutoff, dtype=mstype.float32)
        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions

        tensor1 = Tensor(np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32), dtype=mstype.float32)
        self.frequencies = Parameter(tensor1, name="frequencies", requires_grad=True)

        
    def construct(self, inputs):
        

        d_scaled = inputs * self.inv_cutoff

        # Necessary for proper broadcasting behaviour

        d_scaled = ops.expand_dims(d_scaled, -1)

        d_cutoff = self.envelope(d_scaled)

        return d_cutoff * ops.sin(self.frequencies * d_scaled)

    
def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return np.sqrt(np.pi/(2*r)) * sp.jv(n+0.5, r)


def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1, dtype=np.float32) * np.pi
    points = np.arange(1, k + n, dtype=np.float32) * np.pi
    racines = np.zeros(k + n - 1, dtype=np.float32)
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols('x')

    f = [sym.sin(x)/x]
    a = sym.sin(x)/x
    for i in range(1, n):
        b = sym.diff(a, x)/x
        f += [sym.simplify(b*(-x)**i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    """

    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5*Jn(zeros[order, i], order+1)**2]
        normalizer_tmp = 1/np.array(normalizer_tmp, dtype=np.float32)**0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sym.simplify(normalizer[order]
                                            [i]*f[order].subs(x, zeros[order, i]*x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def associated_legendre_polynomials(l, zero_m_only=True):
    """
    Computes sympy formulas of the associated legendre polynomials up to order l (excluded).
    """
    z = sym.symbols('z')
    P_l_m = [[0]*(j+1) for j in range(l)]

    P_l_m[0][0] = 1
    if l > 0:
        P_l_m[1][0] = z

        for j in range(2, l):
            P_l_m[j][0] = sym.simplify(
                ((2*j-1)*z*P_l_m[j-1][0] - (j-1)*P_l_m[j-2][0])/j)
        if not zero_m_only:
            for i in range(1, l):
                P_l_m[i][i] = sym.simplify((1-2*i)*P_l_m[i-1][i-1])
                if i + 1 < l:
                    P_l_m[i+1][i] = sym.simplify((2*i+1)*z*P_l_m[i][i])
                for j in range(i + 2, l):
                    P_l_m[j][i] = sym.simplify(
                        ((2*j-1) * z * P_l_m[j-1][i] - (i+j-1) * P_l_m[j-2][i]) / (j - i))

    return P_l_m


def sph_harm_prefactor(l, m):
    """
    Computes the constant pre-factor for the spherical harmonic of degree l and order m
    input:
    l: int, l>=0
    m: int, -l<=m<=l
    """
    return ((2*l+1) * np.math.factorial(l-abs(m)) / (4*np.pi*np.math.factorial(l+abs(m))))**0.5


def real_sph_harm(l, zero_m_only=True, spherical_coordinates=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to order l (excluded).
    Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.
    """
    if not zero_m_only:
        S_m = [0]
        C_m = [1]
        for i in range(1, l):
            x = sym.symbols('x')
            y = sym.symbols('y')
            S_m += [x*S_m[i-1] + y*C_m[i-1]]
            C_m += [x*C_m[i-1] - y*S_m[i-1]]

    P_l_m = associated_legendre_polynomials(l, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))

    Y_func_l_m = [['0']*(2*j + 1) for j in range(l)]
    for i in range(l):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m


class SphericalBasisLayer(nn.Cell):
    

    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent=5):

        super(SphericalBasisLayer, self).__init__()
        
        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical

        self.inv_cutoff = Tensor(1 / cutoff, dtype=mstype.float32)
        self.envelope = Envelope(envelope_exponent)

        # retrieve formulas
        self.bessel_formulas = bessel_basis(num_spherical, num_radial)
        self.sph_harm_formulas = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []
        self.gather = ops.Gather()
        # convert to functions
        x = sym.symbols('x')
        theta = sym.symbols('theta')
        for i in range(num_spherical):
            if i == 0:
                first_sph = sym.lambdify([theta], self.sph_harm_formulas[i][0])(0)
                self.sph_funcs.append(lambda array: np.zeros_like(array, dtype=np.float32) + first_sph)
                #self.sph_funcs.append(lambda tensor: ops.ZerosLike(tensor) + first_sph)
            else:
                self.sph_funcs.append(sym.lambdify([theta], self.sph_harm_formulas[i][0]))

            for j in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], self.bessel_formulas[i][j]))


        
    def construct(self, inputs):
        d, Angles, id_expand_kj = inputs
        
        d_scaled = d * self.inv_cutoff
        
        x = d_scaled
        #rbf = [Tensor(f(d_scaled.asnumpy()), dtype=mstype.float16) for f in self.bessel_funcs]
        rbf = [ops.div(1.41421360172718*ops.sin(3.14159274101257*x), x), ops.div(1.41421360172719*ops.sin(6.28318548202515*x), x), ops.div(1.41421356595183*ops.sin(9.42477798461914*x), x), ops.div(1.41421360172725*ops.sin(12.5663709640503*x), x), ops.div(1.41421353733157*ops.sin(15.7079629898071*x), x), ops.div(1.41421356595183*ops.sin(18.8495559692383*x), x), ops.div(ops.add(ops.mul(-1.44881182146373*x, ops.cos(4.49340963363647*x)), 0.322430390191518*ops.sin(4.49340963363647*x)), ops.square(x)), ops.div(ops.add(ops.mul(-1.42601268061811*x, ops.cos(7.7252516746521*x)), 0.184591096921426*ops.sin(7.7252516746521*x)), ops.square(x)), ops.div(ops.add(ops.mul(-1.42014812031817*x, ops.cos(10.9041213989258*x)), 0.130239573493567*ops.sin(10.9041213989258*x)), ops.square(x)), ops.div(ops.add(ops.mul(-1.41778280396269*x, ops.cos(14.0661935806274*x)), 0.100793636589455*ops.sin(14.0661935806274*x)), ops.square(x)), ops.div(ops.add(ops.mul(-1.41659585719745*x, ops.cos(17.2207546234131*x)), 0.0822609629006305*ops.sin(17.2207546234131*x)), ops.square(x)), ops.div(ops.add(ops.mul(-1.41591653329072*x, ops.cos(20.3713035583496*x)), 0.0695054456989023*ops.sin(20.3713035583496*x)), ops.square(x)), 
               ops.div(ops.add(ops.sub(ops.mul(-1.48220819228598*ops.pow(x, 2), ops.sin(5.76345920562744*x)), ops.mul(0.771520092051013*x, ops.cos(5.76345920562744*x))), 0.133864067485322*ops.sin(5.76345920562744*x)), ops.pow(x, 3)), 
               ops.div(ops.add(ops.sub(ops.mul(-1.44054353387536*ops.pow(x, 2), ops.sin(9.09501171112061*x)), ops.mul(0.475164929841923*x, ops.cos(9.09501171112061*x))), 0.0522445649257308*ops.sin(9.09501171112061*x)), ops.pow(x, 3)), 
               ops.div(ops.add(ops.sub(ops.mul(-1.42838784692185*ops.pow(x, 2), ops.sin(12.322940826416*x)), ops.mul(0.347738709544047*x, ops.cos(12.322940826416*x))), 0.0282188086790629*ops.sin(12.322940826416*x)), ops.pow(x, 3)), 
               ops.div(ops.add(ops.sub(ops.mul(-1.42310835777527*ops.pow(x, 2), ops.sin(15.5146026611328*x)), ops.mul(0.27518107724546*x, ops.cos(15.5146026611328*x))), 0.0177369078187767*ops.sin(15.5146026611328*x)), ops.pow(x, 3)), 
               ops.div(ops.add(ops.sub(ops.mul(-1.42032571184861*ops.pow(x, 2), ops.sin(18.6890354156494*x)), ops.mul(0.227993421853001*x, ops.cos(18.6890354156494*x))), 0.0121993145597064*ops.sin(18.6890354156494*x)), ops.pow(x, 3)), 
               ops.div(ops.add(ops.sub(ops.mul(-1.41867612061334*ops.pow(x, 2), ops.sin(21.853874206543*x)), ops.mul(0.194749375859672*x, ops.cos(21.853874206543*x))), 0.00891143483389159*ops.sin(21.853874206543*x)), ops.pow(x, 3)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.mul(1.51240002169845*ops.pow(x, 3), ops.cos(6.9879322052002*x)), ops.mul(1.2985815923397*ops.pow(x, 2), ops.sin(6.9879322052002*x))), ops.mul(0.464580062530277*x, ops.cos(6.9879322052002*x))), 0.0664831954415001*ops.sin(6.9879322052002*x)), ops.pow(x, 4)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.mul(1.45553308963651*ops.pow(x, 3), ops.cos(10.4171180725098*x)), ops.mul(0.838350729734507*ops.pow(x, 2), ops.sin(10.4171180725098*x))), ops.mul(0.20119545633904*x, ops.cos(10.4171180725098*x))), 0.0193139268402827*ops.sin(10.4171180725098*x)), ops.pow(x, 4)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.mul(1.43755963546005*ops.pow(x, 3), ops.cos(13.6980228424072*x)), ops.mul(0.629679035580038*ops.pow(x, 2), ops.sin(13.6980228424072*x))), ops.mul(0.114921518752078*x, ops.cos(13.6980228424072*x))), 0.0083896428027771*ops.sin(13.6980228424072*x)), ops.pow(x, 4)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.mul(1.42934084814112*ops.pow(x, 3), ops.cos(16.9236221313477*x)), ops.mul(0.506749974815457*ops.pow(x, 2), ops.sin(16.9236221313477*x))), ops.mul(0.074858380032724*x, ops.cos(16.9236221313477*x))), 0.00442330722417063*ops.sin(16.9236221313477*x)), ops.pow(x, 4)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.mul(1.42484891172507*ops.pow(x, 3), ops.cos(20.1218070983887*x)), ops.mul(0.42486708219338*ops.pow(x, 2), ops.sin(20.1218070983887*x))), ops.mul(0.0527868943524714*x, ops.cos(20.1218070983887*x))), 0.00262336747859483*ops.sin(20.1218070983887*x)), ops.pow(x, 4)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.mul(1.42211242909534*ops.pow(x, 3), ops.cos(23.3042469024658*x)), ops.mul(0.366142472240508*ops.pow(x, 2), ops.sin(23.3042469024658*x))), ops.mul(0.039278513673163*x, ops.cos(23.3042469024658*x))), 0.00168546590831935*ops.sin(23.3042469024658*x)), ops.pow(x, 4)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.mul(1.53965371388729*ops.pow(x, 4), ops.sin(8.18256187438965*x)), ops.mul(1.8816279516397*ops.pow(x, 3), ops.cos(8.18256187438965*x))), ops.mul(1.03480131435123*ops.pow(x, 2), ops.sin(8.18256187438965*x))), ops.mul(0.29508318265335*x, ops.cos(8.18256187438965*x))), 0.0360624444010527*ops.sin(8.18256187438965*x)), ops.pow(x, 5)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.mul(1.47025596792433*ops.pow(x, 4), ops.sin(11.7049074172974*x)), ops.mul(1.25610217621337*ops.pow(x, 3), ops.cos(11.7049074172974*x))), ops.mul(0.482913669578201*ops.pow(x, 2), ops.sin(11.7049074172974*x))), ops.mul(0.0962671913734207*x, ops.cos(11.7049074172974*x))), 0.00822451540549208*ops.sin(11.7049074172974*x)), ops.pow(x, 5)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.mul(1.44705837727913*ops.pow(x, 4), ops.sin(15.0396642684937*x)), ops.mul(0.962161356427715*ops.pow(x, 3), ops.cos(15.0396642684937*x))), ops.mul(0.287887151377108*ops.pow(x, 2), ops.sin(15.0396642684937*x))), ops.mul(0.044664340543412*x, ops.cos(15.0396642684937*x))), 0.00296976978648244*ops.sin(15.0396642684937*x)), ops.pow(x, 5)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.mul(1.43603814087772*ops.pow(x, 4), ops.sin(18.3012561798096*x)), ops.mul(0.784666433150089*ops.pow(x, 3), ops.cos(18.3012561798096*x))), ops.mul(0.192937518303847*ops.pow(x, 2), ops.sin(18.3012561798096*x))), ops.mul(0.0245987236223508*x, ops.cos(18.3012561798096*x))), 0.00134410028364549*ops.sin(18.3012561798096*x)), ops.pow(x, 5)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.mul(1.42984401711803*ops.pow(x, 4), ops.sin(21.5254173278809*x)), ops.mul(0.664258441700927*ops.pow(x, 3), ops.cos(21.5254173278809*x))), ops.mul(0.138866668279757*ops.pow(x, 2), ops.sin(21.5254173278809*x))), ops.mul(0.0150530055260025*x, ops.cos(21.5254173278809*x))), 0.000699313063097041*ops.sin(21.5254173278809*x)), ops.pow(x, 5)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.mul(1.42598923008548*ops.pow(x, 4), ops.sin(24.7275657653809*x)), ops.mul(0.576679986867893*ops.pow(x, 3), ops.cos(24.7275657653809*x))), ops.mul(0.104946033326849*ops.pow(x, 2), ops.sin(24.7275657653809*x))), ops.mul(0.00990287843478219*x, ops.cos(24.7275657653809*x))), 0.000400479308345281*ops.sin(24.7275657653809*x)), ops.pow(x, 5)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.mul(-1.56444176824918*ops.pow(x, 5), ops.cos(9.35581207275391*x)), ops.mul(2.50824047568009*ops.pow(x, 4), ops.sin(9.35581207275391*x))), ops.mul(1.87666053926973*ops.pow(x, 3), ops.cos(9.35581207275391*x))), ops.mul(0.802350677707588*ops.pow(x, 2), ops.sin(9.35581207275391*x))), ops.mul(0.192959094390048*x, ops.cos(9.35581207275391*x))), 0.0206245158506321*ops.sin(9.35581207275391*x)), ops.pow(x, 6)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.mul(-1.48446325845568*ops.pow(x, 5), ops.cos(12.9665298461914*x)), ops.mul(1.71726353472865*ops.pow(x, 4), ops.sin(12.9665298461914*x))), ops.mul(0.927067217342765*ops.pow(x, 3), ops.cos(12.9665298461914*x))), ops.mul(0.285987763369108*ops.pow(x, 2), ops.sin(12.9665298461914*x))), ops.mul(0.0496256496698302*x, ops.cos(12.9665298461914*x))), 0.00382721130930852*ops.sin(12.9665298461914*x)), ops.pow(x, 6)),
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.mul(-1.45659317306845*ops.pow(x, 5), ops.cos(16.3547096252441*x)), ops.mul(1.33593919407179*ops.pow(x, 4), ops.sin(16.3547096252441*x))), ops.mul(0.571797028060223*ops.pow(x, 3), ops.cos(16.3547096252441*x))), ops.mul(0.13984889763561*ops.pow(x, 2), ops.sin(16.3547096252441*x))), ops.mul(0.0192397191323062*x, ops.cos(16.3547096252441*x))), 0.00117640236807439*ops.sin(16.3547096252441*x)), ops.pow(x, 6)),
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.mul(-1.44295592066705*ops.pow(x, 5), ops.cos(19.6531524658203*x)), ops.mul(1.10131638410929*ops.pow(x, 4), ops.sin(19.6531524658203*x))), ops.mul(0.392263516103712*ops.pow(x, 3), ops.cos(19.6531524658203*x))), ops.mul(0.079837271254251*ops.pow(x, 2), ops.sin(19.6531524658203*x))), ops.mul(0.00914020591019553*x, ops.cos(19.6531524658203*x))), 0.000465075815500423*ops.sin(19.6531524658203*x)), ops.pow(x, 6)),
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.mul(-1.43511786437635*ops.pow(x, 5), ops.cos(22.9045505523682*x)), ops.mul(0.939846774833116*ops.pow(x, 4), ops.sin(22.9045505523682*x))), ops.mul(0.287232330046817*ops.pow(x, 3), ops.cos(22.9045505523682*x))), ops.mul(0.0501616182147035*ops.pow(x, 2), ops.sin(22.9045505523682*x))), ops.mul(0.00492756409801779*x, ops.cos(22.9045505523682*x))), 0.000215134721231555*ops.sin(22.9045505523682*x)), ops.pow(x, 6)),
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.mul(-1.43015328644677*ops.pow(x, 5), ops.cos(26.1277503967285*x)), ops.mul(0.821054203709311*ops.pow(x, 4), ops.sin(26.1277503967285*x))), ops.mul(0.219972226414288*ops.pow(x, 3), ops.cos(26.1277503967285*x))), ops.mul(0.0336764127143271*ops.pow(x, 2), ops.sin(26.1277503967285*x))), ops.mul(0.00290005559057712*x, ops.cos(26.1277503967285*x))), 0.000110995227164304*ops.sin(26.1277503967285*x)), ops.pow(x, 6)),
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.sub(ops.mul(-1.58717984397985*ops.pow(x, 6), ops.sin(10.5128355026245*x)), ops.mul(3.1704839969441*ops.pow(x, 5), ops.cos(10.5128355026245*x))), ops.mul(3.01582194085753*ops.pow(x, 4), ops.sin(10.5128355026245*x))), ops.mul(1.72122275104826*ops.pow(x, 3), ops.cos(10.5128355026245*x))), ops.mul(0.613971874174155*ops.pow(x, 2), ops.sin(10.5128355026245*x))), ops.mul(0.128484662662698*x, ops.cos(10.5128355026245*x))), 0.0122216943878387*ops.sin(10.5128355026245*x)), ops.pow(x, 7)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.sub(ops.mul(-1.49808440567462*ops.pow(x, 6), ops.sin(14.2073926925659*x)), ops.mul(2.21432413391576*ops.pow(x, 5), ops.cos(14.2073926925659*x))), ops.mul(1.55857178148839*ops.pow(x, 4), ops.sin(14.2073926925659*x))), ops.mul(0.658208785474305*ops.pow(x, 3), ops.cos(14.2073926925659*x))), ops.mul(0.173732295498539*ops.pow(x, 2), ops.sin(14.2073926925659*x))), ops.mul(0.0269022654872332*x, ops.cos(14.2073926925659*x))), 0.0018935399386342*ops.sin(14.2073926925659*x)), ops.pow(x, 7)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.sub(ops.mul(-1.46601645543643*ops.pow(x, 6), ops.sin(17.6479740142822*x)), ops.mul(1.74446911238934*ops.pow(x, 5), ops.cos(17.6479740142822*x))), ops.mul(0.988481233583851*ops.pow(x, 4), ops.sin(17.6479740142822*x))), ops.mul(0.33606619075387*ops.pow(x, 3), ops.cos(17.6479740142822*x))), ops.mul(0.0714103621360227*ops.pow(x, 2), ops.sin(17.6479740142822*x))), ops.mul(0.00890203014646945*x, ops.cos(17.6479740142822*x))), 0.000504422215222279*ops.sin(17.6479740142822*x)), ops.pow(x, 7)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.sub(ops.mul(-1.44995111708105*ops.pow(x, 6), ops.sin(20.9834632873535*x)), ops.mul(1.4510937990419*ops.pow(x, 5), ops.cos(20.9834632873535*x))), ops.mul(0.691541610252895*ops.pow(x, 4), ops.sin(20.9834632873535*x))), ops.mul(0.197739029286842*ops.pow(x, 3), ops.cos(20.9834632873535*x))), ops.mul(0.0353383685843968*ops.pow(x, 2), ops.sin(20.9834632873535*x))), ops.mul(0.0037050323781646*x, ops.cos(20.9834632873535*x))), 0.000176569154835255*ops.sin(20.9834632873535*x)), ops.pow(x, 7)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.sub(ops.mul(-1.44054685504282*ops.pow(x, 6), ops.sin(24.262767791748*x)), ops.mul(1.24682741126459*ops.pow(x, 5), ops.cos(24.262767791748*x))), ops.mul(0.513885069488506*ops.pow(x, 4), ops.sin(24.262767791748*x))), ops.mul(0.12707991286879*ops.pow(x, 3), ops.cos(24.262767791748*x))), ops.mul(0.0196411916953696*ops.pow(x, 2), ops.sin(24.262767791748*x))), ops.mul(0.0017809436293789*x, ops.cos(24.262767791748*x))), 7.34023275771782e-5*ops.sin(24.262767791748*x)), ops.pow(x, 7)), 
              ops.div(ops.add(ops.sub(ops.sub(ops.add(ops.add(ops.sub(ops.mul(-1.43450193707163*ops.pow(x, 6), ops.sin(27.5078678131104*x)), ops.mul(1.09512452521481*ops.pow(x, 5), ops.cos(27.5078678131104*x))), ops.mul(0.398113198978247*ops.pow(x, 4), ops.sin(27.5078678131104*x))), ops.mul(0.0868362175541293*ops.pow(x, 3), ops.cos(27.5078678131104*x))), ops.mul(0.0118379155389421*ops.pow(x, 2), ops.sin(27.5078678131104*x))), ops.mul(0.00094676237222793*x, ops.cos(27.5078678131104*x))), 3.44178755932766e-5*ops.sin(27.5078678131104*x)), ops.pow(x, 7))]
        rbf = ops.stack(rbf, axis=1)
        
        d_cutoff = self.envelope(d_scaled)
        
        rbf_env = d_cutoff[:, None] * rbf
        
        rbf_env = self.gather(rbf_env, id_expand_kj, 0)
        
        #cbf = [Tensor(f(Angles.asnumpy()), dtype=mstype.float16) for f in self.sph_funcs]
        theta = Angles
        cbf = [0.282094791773878*ops.pow(theta, 0), 0.48860251190292*ops.cos(theta), 0.94617469575756*ops.pow(ops.cos(theta), 2) - 0.31539156525252, ops.mul((1.86588166295058*ops.pow(ops.cos(theta), 2) - 1.11952899777035), ops.cos(theta)), 
               ops.sub(3.70249414203215*ops.pow(ops.cos(theta), 4), 3.17356640745613*ops.pow(ops.cos(theta), 2)) + 0.317356640745613, 
               ops.mul(ops.sub(7.36787031456569*ops.pow(ops.cos(theta), 4), 8.18652257173965*ops.pow(ops.cos(theta), 2)) + 1.75425483680135, ops.cos(theta)),
              ops.add(ops.sub(14.6844857238222*ops.pow(ops.cos(theta), 6), 20.024298714303*ops.pow(ops.cos(theta), 4)), 6.67476623810098*ops.pow(ops.cos(theta), 2)) - 0.317846011338142]
        
        cbf = ops.stack(cbf, axis=1)
        
        cbf = cbf.repeat(self.num_radial, axis=1)

        return rbf_env * cbf

    
class Swish(nn.Cell):
    """
    Swish activation function,
    from Ramachandran, Zopf, Le 2017. "Searching for Activation Functions"
    """

    def __init__(self):
        super(Swish, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
      
        
    def construct(self, x):

        return x*self.sigmoid(x)



class EmbeddingBlock(nn.Cell):
    

    def __init__(self, emb_size, activation=None):

        super(EmbeddingBlock, self).__init__()
        
        self.emb_size = emb_size

        # Atom embeddings: We go up to Pu (94). Use 95 dimensions because of 0-based indexing
        tensor1 = initializer(Uniform(scale=np.sqrt(3.0)), shape=[95, self.emb_size], dtype=mstype.float32)
        self.embeddings = Parameter(tensor1, name="embeddings", requires_grad=True)
        
        # TODO: weight_init = ?
        self.dense_rbf = nn.Dense(6, self.emb_size, weight_init='orthogonal', has_bias=True, activation=activation)    # in_channel=6?
        self.dense = nn.Dense(384, self.emb_size, weight_init='orthogonal', has_bias=True, activation=activation)      # in_channel=384?
        self.gather = ops.Gather()
        

        
    def construct(self, inputs):
        
        Z, rbf, idnb_i, idnb_j = inputs

        rbf = self.dense_rbf(rbf)

        Z_i = self.gather(Z, idnb_i, 0)
        Z_j = self.gather(Z, idnb_j, 0)

        x_i = self.gather(self.embeddings, Z_i, 0)
        x_j = self.gather(self.embeddings, Z_j, 0)

        x = ops.concat([x_i, x_j, rbf], axis=-1)
        x = self.dense(x)
        return x

class OutputPPBlock(nn.Cell):
    

    def __init__(self, emb_size, out_emb_size, num_dense, num_targets=12,
                 activation=None, output_init='zeros'):

        super(OutputPPBlock, self).__init__()
        
        self.dense_rbf = nn.Dense(6, emb_size, weight_init='orthogonal', has_bias=False)
        self.up_projection = nn.Dense(128, out_emb_size, weight_init='orthogonal', has_bias=False)                                  

        self.dense_layers = []
        for i in range(num_dense):
            self.dense_layers.append(nn.Dense(256, out_emb_size, weight_init='orthogonal', has_bias=True, activation=activation))
            #self.dense_layers.append(nn.Dense(256, out_emb_size, weight_init='orthogonal', has_bias=True, activation=activation).to_float(mstype.float16)) 
        self.dense_final = nn.Dense(256, num_targets, weight_init='orthogonal', has_bias=False)    

    def construct(self, inputs):
        
        x, rbf, idnb_i, n_atoms = inputs
        
        g = self.dense_rbf(rbf)
        x = g * x
        x = ops.unsorted_segment_sum(x, idnb_i, n_atoms)

        x = self.up_projection(x)

        for layer in self.dense_layers:
            x = layer(x)
        x = self.dense_final(x)
        return x
    
class ResidualLayer(nn.Cell):
    

    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):

        super(ResidualLayer, self).__init__()
        self.dense_1 = nn.Dense(128, units, weight_init='orthogonal', bias_init=bias_initializer, has_bias=use_bias, activation=activation)
        self.dense_2 = nn.Dense(128, units, weight_init='orthogonal', bias_init=bias_initializer, has_bias=use_bias, activation=activation)

        
    def construct(self, inputs):
        
        x = inputs + self.dense_2(self.dense_1(inputs))
        return x
    
    
class InteractionPPBlock(nn.Cell):
    

    def __init__(self, emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip, activation=None):

        super(InteractionPPBlock, self).__init__()

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf1 = nn.Dense(6, basis_emb_size, weight_init='orthogonal', has_bias=False)
        self.dense_rbf2 = nn.Dense(8, emb_size, weight_init='orthogonal', has_bias=False)
        self.dense_sbf1 = nn.Dense(42, basis_emb_size, weight_init='orthogonal', has_bias=False)
        self.dense_sbf2 = nn.Dense(8, int_emb_size, weight_init='orthogonal', has_bias=False)

        # Dense transformations of input messages
        self.dense_ji = nn.Dense(128, emb_size, weight_init='orthogonal', has_bias=True, activation=activation)
        self.dense_kj = nn.Dense(128, emb_size, weight_init='orthogonal', has_bias=True, activation=activation)

        # Embedding projections for interaction triplets
        self.down_projection = nn.Dense(128, int_emb_size, weight_init='orthogonal', has_bias=False, activation=activation)
        self.up_projection = nn.Dense(64, emb_size, weight_init='orthogonal', has_bias=False, activation=activation)

        # Residual layers before skip connection
        self.layers_before_skip = []
        for i in range(num_before_skip):
            self.layers_before_skip.append(ResidualLayer(emb_size, activation=activation, use_bias=True))
            #self.layers_before_skip.append(ResidualLayer(emb_size, activation=activation, use_bias=True).to_float(mstype.float16))
        self.final_before_skip = nn.Dense(128, emb_size, weight_init='orthogonal', has_bias=True, activation=activation)

        # Residual layers after skip connection
        self.layers_after_skip = []
        for i in range(num_after_skip):
            self.layers_after_skip.append(ResidualLayer(emb_size, activation=activation, use_bias=True))
            #self.layers_after_skip.append(ResidualLayer(emb_size, activation=activation, use_bias=True).to_float(mstype.float16))
        self.gather = ops.Gather()

        
    def construct(self, inputs):
        x, rbf, sbf, id_expand_kj, id_reduce_ji = inputs
        num_interactions = x.shape[0]

        # Initial transformation
        x_ji = self.dense_ji(x)
        x_kj = self.dense_kj(x)

        # Transform via Bessel basis

        rbf = self.dense_rbf1(rbf)

        rbf = self.dense_rbf2(rbf)

        x_kj = x_kj * rbf

        # Down-project embeddings and generate interaction triplet embeddings
        x_kj = self.down_projection(x_kj)
        x_kj = self.gather(x_kj, id_expand_kj, 0)

        # Transform via 2D spherical basis
        sbf = self.dense_sbf1(sbf)
        sbf = self.dense_sbf2(sbf)
        x_kj = x_kj * sbf

        # Aggregate interactions and up-project embeddings

        x_kj = mindspore.numpy.where(ops.IsNan()(x_kj), ops.ZerosLike()(x_kj), x_kj)

        x_kj = ops.unsorted_segment_sum(x_kj, id_reduce_ji, num_interactions)

        x_kj = self.up_projection(x_kj)

        # Transformations before skip connection
        x2 = x_ji + x_kj
        for layer in self.layers_before_skip:
            x2 = layer(x2)
        x2 = self.final_before_skip(x2)

        # Skip connection
        x = x + x2

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            x = layer(x)
        return x
    
class DimenetPP(nn.Cell):

    def __init__(self, emb_size=128, out_emb_size=256, int_emb_size=64, basis_emb_size=8,
            num_blocks=4, num_spherical=7, num_radial=6,
            cutoff=5.0, envelope_exponent=5, num_before_skip=1,
            num_after_skip=2, num_dense_output=3, num_targets=1,
            activation=Swish(), extensive=True):
        
            #num_targets=12 to process all predicitons simultaneously? 
        
        super().__init__()
        
        self.num_blocks = num_blocks
        self.extensive = extensive

        # Cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        self.sbf_layer = SphericalBasisLayer(num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        
        # Embedding and first output block
        self.output_blocks = []
        self.emb_block = EmbeddingBlock(emb_size, activation=activation)
        self.output_blocks.append(OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_targets, activation=activation))
        #self.output_blocks.append(OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_targets, activation=activation).to_float(mstype.float16))
        # Interaction and remaining output blocks
        self.int_blocks = []
        for i in range(num_blocks):
            self.int_blocks.append(InteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip, activation=activation))
            self.output_blocks.append(OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_targets, activation=activation))
            #self.int_blocks.append(InteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip, activation=activation).to_float(mstype.float16))
            #self.output_blocks.append(OutputPPBlock(emb_size, out_emb_size, num_dense_output, num_targets, activation=activation).to_float(mstype.float16))
    def calculate_interatomic_distances(self, R, idx_i, idx_j):
        Ri = ops.Gather()(R, idx_i, 0)
        Rj = ops.Gather()(R, idx_j, 0)

        # ReLU prevents negative numbers in sqrt
        Dij = ops.sqrt(nn.ReLU()(((Ri - Rj)**2).sum(-1)))

        return Dij
    
    def calculate_neighbor_angles(self, R, id3_i, id3_j, id3_k):
        """Calculate angles for neighboring atom triplets"""
        Ri = ops.Gather()(R, id3_i, 0)
        Rj = ops.Gather()(R, id3_j, 0)
        Rk = ops.Gather()(R, id3_k, 0)
        R1 = Rj - Ri
        R2 = Rk - Rj
        x = (R1 * R2).sum(-1)
        y = mindspore.numpy.cross(R1, R2)
        y = ops.norm(y, axis=-1)
        angle = ops.atan2(y, x)
        return angle
    
    
    def construct(self, Z, R, batch_seg, idnb_i, idnb_j, id_expand_kj, id_reduce_ji, id3dnb_i, id3dnb_j, id3dnb_k):
        n_atoms = Z.shape[0]
        
        # Calculate distances
        Dij = self.calculate_interatomic_distances(R, idnb_i, idnb_j)

        rbf = self.rbf_layer(Dij)

        rbf = mindspore.numpy.where(ops.IsNan()(rbf), ops.ZerosLike()(rbf), rbf)
        # Calculate angles
        Anglesijk = self.calculate_neighbor_angles(R, id3dnb_i, id3dnb_j, id3dnb_k)
        

        
        sbf = self.sbf_layer([Dij, Anglesijk, id_expand_kj])
        sbf = mindspore.numpy.where(ops.IsNan()(sbf), ops.ZerosLike()(sbf), sbf)
        
        # Embedding block
        x = self.emb_block([Z, rbf, idnb_i, idnb_j])
        

        
        P = self.output_blocks[0]([x, rbf, idnb_i, n_atoms])

        

        
        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.int_blocks[i]([x, rbf, sbf, id_expand_kj, id_reduce_ji])
            P += self.output_blocks[i+1]([x, rbf, idnb_i, n_atoms])

        if self.extensive:
            P = ops.unsorted_segment_sum(P, batch_seg, 33)    # batch_size=32
        #else:
        #    P = tf.math.segment_mean(P, batch_seg)

        return P





