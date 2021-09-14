#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:46:21 2021

@author: nickolai
"""

import cr3bp
from scipy.integrate import solve_ivp
import numpy as np


def test_initial_velocity():
    EM = cr3bp.EarthMoon
    # relative to Lagrange point in m
    initial_position = [EM.L1 + 1000/EM.l, 2000/EM.l]
    iv = cr3bp.initial_velocity(initial_position, EM.L1, EM.mu)
    assert np.isclose(iv[0], 3.3864846085849844e-06)
    assert np.isclose(iv[1], -2.1781128578193083e-05)


def test_system_creation():
    mass_sun = 1.989e30  # kg
    mass_jupiter = 1.898e27  # kg
    sun_jupiter_distance = 778.34e9  # m
    SJ = cr3bp.System(mass_sun, mass_jupiter, sun_jupiter_distance)
    assert np.isclose(SJ.mu, mass_jupiter/(mass_sun + mass_jupiter))
    assert np.isclose(SJ.total_mass, mass_sun + mass_jupiter)
    assert np.isclose(SJ.theta_dot, np.sqrt(cr3bp.System.G*(mass_sun+mass_jupiter)
                                            / (sun_jupiter_distance**3)))
    assert np.isclose(SJ.seconds, 1/SJ.theta_dot)
    assert np.isclose(SJ.L2, 831.90e9/sun_jupiter_distance)
    assert np.isclose(SJ.L3, -778.65e9/sun_jupiter_distance)
    assert np.isclose(SJ.L4[0], 0.5)
    assert np.isclose(SJ.L4[1], np.sqrt(3)/2)
    assert np.isclose(SJ.L5[0], 0.5)
    assert np.isclose(SJ.L5[1], -np.sqrt(3)/2)


def test_dimensional_EOMS():
    tf = 3600*24*7  # s
    IC = [cr3bp.SunEarth.L1*cr3bp.SunEarth.l + 1000000, 0, 0, 0, 0, 0]  # m
    # We set the absolute tolerance to be at the micrometer level and keep the rtol small enough so that
    # the atol dominates
    solution = solve_ivp(cr3bp.DimensionalEOMs, [0, tf], IC, args=[cr3bp.SunEarth], atol=1e-6, rtol=1e-12)
    final_state = solution.y.T[-1]
    assert np.isclose(final_state, [1.48109206e+11, -5.32205049e+03,  0.00000000e+00,
                                    2.21611267e-01, -2.64256808e-02,  0.00000000e+00]).all()


def test_EOM_constructor():
    EM = cr3bp.EarthMoon
    eoms = cr3bp.EOMConstructor(EM.mu)
    tf = 3600*24*7 / EM.seconds  # s*
    IC = [EM.L1 + 1000/EM.l, 0, 0, 0, 0, 0]  # m*
    # We set the absolute tolerance to be at the micrometer level and keep the rtol small enough so that
    # the atol dominates
    solution = solve_ivp(eoms, [0, tf], IC, atol=0.000001/EM.l, rtol=2.3e-14)
    final_state = solution.y.T[-1]
    assert np.isclose(final_state, [8.37055164e-01, -8.15798446e-05,  0.00000000e+00,
                                    5.13004292e-04, -2.39580318e-04, 0.00000000e+00]).all()


def test_STM():
    EM = cr3bp.EarthMoon
    eoms = cr3bp.EOMConstructor(EM.mu, STM=True)
    tf = 3600*24*7 / EM.seconds  # s*
    IC = [EM.L1 + 1000/EM.l, 0, 0, 0, 0, 0]  # m*
    IC = np.concatenate((IC, np.eye(6).reshape(36)))
    # We set the absolute tolerance to be at the micrometer level and keep the rtol small enough so that
    # the atol dominates
    solution = solve_ivp(eoms, [0, tf], IC, atol=0.000001/EM.l, rtol=2.3e-14)
    final_state = solution.y.T[-1]
    assert np.isclose(final_state, [8.37055163e-01, -8.15798392e-05,  0.00000000e+00,
                                    5.13004258e-04, -2.39580302e-04,  0.00000000e+00,
                                    6.75100337e+01, -1.15247348e+01,  0.00000000e+00,  1.74575908e+01,  8.16014216e+00,  0.00000000e+00,  # noqa: E501
                                   -3.13594523e+01,  4.49401735e+00,  0.00000000e+00, -8.15294382e+00, -3.99485808e+00,  0.00000000e+00,  # noqa: E501, E128
                                    0.00000000e+00,  0.00000000e+00, -8.69308229e-01,  0.00000000e+00,  0.00000000e+00, -2.17501858e-01,  # noqa: E501
                                    1.97373618e+02, -3.38753270e+01,  0.00000000e+00,  5.12337262e+01,  2.34112148e+01,  0.00000000e+00,  # noqa: E501
                                   -9.20938835e+01,  1.65711396e+01,  0.00000000e+00, -2.33685040e+01, -1.18119426e+01,  0.00000000e+00,  # noqa: E501, E128
                                    0.00000000e+00,  0.00000000e+00,  1.12153698e+00,  0.00000000e+00,  0.00000000e+00, -8.69730203e-01]).all()  # noqa: E501


def test_conversion():
    # run dimensionless first
    EM = cr3bp.EarthMoon
    eoms = cr3bp.EOMConstructor(EM.mu)
    tf = 3600*24*7 / EM.seconds  # s*
    IC = [EM.L1 + 1000/EM.l, 0, 0, 0, 0, 0]  # m*
    solution = solve_ivp(eoms, [0, tf], IC, atol=1e-12, rtol=1e-12)
    final_state_dimensionless = solution.y.T[-1]

    # then run dimensional
    tf = tf * EM.seconds  # s
    IC = EM.convert_to_dimensional_state(IC)  # m
    solution = solve_ivp(cr3bp.DimensionalEOMs, [0, tf], IC, args=[EM], atol=1e-6, rtol=1e-12)
    final_state_dimensional = solution.y.T[-1]

    # then run converter
    final_state_converted = EM.convert_to_dimensional_state(final_state_dimensionless)
    assert np.isclose(final_state_converted, final_state_dimensional).all()
