"""
This module contains all the noisy quantum gates functions

All the time duration are expressed in units of single-qubit gate time tg of IBM's devices.
"""

from numpy import array, exp, sqrt, matmul, pi, sin, cos, kron
from numpy.random import normal, multivariate_normal
from scipy.linalg import expm
from scipy.integrate import quad
import numpy as np


def bitflip(tm: float, rout: float) -> np.array:
    """
    NOISE GATE FOR BITFLIP (exact solution, unitary matrix)
    
    This function implements the bitflip error on measurements 
    
    Args: 
        tm: measurement time in ns
        rout: readout error
        
    Returns:
          bitflip noise gate
    """
    tg = 35 * 10**(-9)
    Dtm = tm / tg
    e = sqrt(rout/Dtm)
    W = normal(0, sqrt(Dtm)) 
    result = array(
        [[np.cos(e * W), 1J * np.sin(e * W)],
        [1J * np.sin(e * W), np.cos(e * W)]]
    )
    return result


def depolarizing(Dt: float, p: float) -> np.array:
    """
    NOISE GATE FOR DEPOLARIZATION (2 order approximated solution, unitary matrix)
    
    This function implements the single-qubit depolarizing error on idle qubits
    
    Args: 
        Dt: idle time in ns
        p: single-qubit depolarizing error probability
        
    Returns:
          depolarizing noise gate
    """
    tg = 35 * 10**(-9)
    Dt = Dt / tg
    ed = sqrt(p/4)
    W1 = normal(0, sqrt(Dt))
    W2 = normal(0, sqrt(Dt))
    W3 = normal(0, sqrt(Dt))
    X = array([[0,1],[1,0]])
    Y = array([[0,-1J],[1J,0]])
    Z = array([[1,0],[0,-1]])
    I1 = ed * X * W1
    I2 = ed * Y * W2
    I3 = ed * Z * W3
    result = expm(1J * I1 + 1J * I2 + 1J * I3)
    return result


def relaxation(Dt: float, T1: float, T2: float) -> np.array:
    """
    NOISE GATE FOR AMPLITUDE AND PHASE DAMPING (exact solution, non-unitary matrix)
    
    This function implements the single-qubit relaxation error on idle qubits
    
    Args: 
        Dt: idle time in ns
        T1: qubit's amplitude damping time in ns
        T2: qubit's dephasing time in ns
        
    Returns:
          amplitude and phase damping noise gate
    """
    # Constants
    tg = 35 * 10**(-9)
    Dt = Dt / tg

    # Helper function
    def V(Dt) -> float:
        return 1-exp(-e1**2 * Dt)

    # Calculations
    if T1 == 0:
        e1 = 0
    else:
        e1 = sqrt(tg/T1)
        
    if T2 == 0:
        ep = 0
    else:
        e2 = sqrt(tg/T2)
        ep = sqrt((1/2) * (e2**2 - e1**2/2))

    W = normal(0, sqrt(Dt))
    I = normal(0, sqrt(V(Dt)))
    result = array(
        [[exp(1J * ep * W), 1J * I * exp(-1J * ep * W)],
        [0, exp(-e1**2/2 * Dt) * exp(-1J * ep * W)]]
    )
    return result


def Noise_Gate(theta: float, phi: float, p: float, T1: float, T2: float) -> np.array:
    """
    NOISY QUANTUM GATE FOR GENERAL SINGLE-QUBIT GATES OF IBM's DEVICES (2 order approximated solution, non-unitary matrix)
    
    This function implements the single-qubit noisy quantum gate with depolarizing and 
    relaxation errors during the unitary evolution.
    
    Args: 
        theta: angle of rotation on the Bloch sphere
        phi: phase of the drive defining axis of rotation on the Bloch sphere
        p: single-qubit depolarizing error probability
        T1: qubit's amplitude damping time in ns
        T2: qubit's dephasing time in ns
        
    Returns:
          general single-qubit noisy quantum gate
    """
    tg = 35 * 10**(-9)
    
    ed = sqrt(p/4)
    
    if T1 == 0:
        e1 = 0
    else:
        e1 = sqrt(tg/T1)
        
    if T2 == 0:
        ep = 0
    else:
        e2 = sqrt(tg/T2)
    
        ep = sqrt((1/2) * (e2**2 - e1**2/2))
 
    # 1) UNITARY CONTRIBUTION
    U = array(
        [[np.cos(theta/2),- 1J * np.sin(theta/2) * exp(-1J * phi)],
        [- 1J * np.sin(theta/2) * exp(1J * phi),np.cos(theta/2)]]
    )

    # 2) DEPOLARIZATION CONTRIBUTION
    
    # Variances and covariances for depolarization Itô processes depending on X(t)
    
    # Integral of sin**2(theta)
    Vdx_1 = (2*theta - np.sin(2*theta))/(4*theta)

    # Integral of sin**4(theta/2)
    Vdx_2 = (6*theta-8*np.sin(theta)+np.sin(2*theta))/(16*theta)

    #Integral of sin(theta) sin**2(theta/2)
    Covdx_12 = ((np.sin(theta/2))**4)/theta

    #Integral of sin(theta)
    Covdx_1Wdx = (1-np.cos(theta))/theta

    #Integral of sin**2(theta/2)
    Covdx_2Wdx = (theta - np.sin(theta))/(2 * theta)

    meand_x = array([0, 0, 0])
    covd_x = array([[Vdx_1, Covdx_12, Covdx_1Wdx], [Covdx_12, Vdx_2, Covdx_2Wdx], [Covdx_1Wdx, Covdx_2Wdx, 1]])

    # The variance of Wr is 1
    sample_dx = multivariate_normal(meand_x, covd_x,1)
    Idx1 = sample_dx[0,0]
    Idx2 = sample_dx[0,1]
    Wdx = sample_dx[0,2]
    Idx = ed * array([[np.sin(phi)*Idx1,Wdx + (exp(-2*1J*phi)-1)*Idx2],[Wdx + (exp(+2*1J*phi)-1)*Idx2,-np.sin(phi)*Idx1]])

    #Variances and covariances for depolarization Itô processes depending on Y(t)

    #Integral of sin**2(theta)
    Vdy_1 = (2*theta - np.sin(2*theta))/(4*theta)

    #Integral of sin**4(theta/2)
    Vdy_2 = (6*theta-8*np.sin(theta)+np.sin(2*theta))/(16*theta) 

    #Integral of sin(theta) sin**2(theta/2)
    Covdy_12 = ((np.sin(theta/2))**4)/theta 

    #Integral of sin(theta)
    Covdy_1Wdy = (1-np.cos(theta))/theta

    #Integral of sin**2(theta/2)
    Covdy_2Wdy = (theta - np.sin(theta))/(2 * theta)

    meand_y = array([0, 0, 0])
    covd_y = array([[Vdy_1, Covdy_12, Covdy_1Wdy], [Covdy_12, Vdy_2, Covdy_2Wdy],[Covdy_1Wdy, Covdy_2Wdy, 1]])
    
    # The variance of Wr is 1
    sample_dy = multivariate_normal(meand_y, covd_y,1)
    Idy1 = sample_dy[0,0]
    Idy2 = sample_dy[0,1]
    Wdy = sample_dy[0,2]
    Idy = ed * array([[-np.cos(phi)*Idy1, -1J*Wdy + 1J*(exp(-2*1J*phi)+1)*Idy2], [1J*Wdy - 1J*(exp(2*1J*phi)+1)*Idy2, np.cos(phi)*Idy1]])

    # Variances and covariances for depolarization Itô processes depending on Z(t)

    # Integral of cos(theta)**2
    Vdz_1 = (2*theta + np.sin(2*theta))/(4*theta)

    # Integral of sin(theta)**2
    Vdz_2 = (2*theta - np.sin(2*theta))/(4*theta)

    # Integral of sin(theta)*cos(theta)
    Covdz_12 = (np.sin(theta))**2/(2*theta)

    meand_z = array([0,0])
    covd_z = array([[Vdz_1,Covdz_12],[Covdz_12,Vdz_2]])
    
    sample_dz = multivariate_normal(meand_z, covd_z,1)
    Idz1 = sample_dz[0,0]
    Idz2 = sample_dz[0,1]
    Idz = ed * array([[Idz1,-1J * exp(-1J*phi) * Idz2],[1J *exp(1J*phi) * Idz2,-Idz1]])
    
    # 3) RELAXATION CONTRIBUTIONS
    
    # Variances and covariances for relaxation Itô processes depending on sigma_min(t)
    
    # Integral of sin**2(theta)
    Vr1 = (2*theta - np.sin(2*theta))/(4*theta)

    # Integral of sin**4(theta/2)
    Vr2 = (6*theta-8*np.sin(theta)+np.sin(2*theta))/(16*theta)

    # Integral of sin(theta) sin**2(theta/2)
    Covr12 = ((np.sin(theta/2))**4)/theta

    # Integral of sin(theta)
    Covr1Wr = (1-np.cos(theta))/theta

    # Integral of sin**2(theta)
    Covr2Wr = (theta - np.sin(theta))/(2 * theta)

    meanr = array([0, 0, 0])
    covr = array([[Vr1, Covr12, Covr1Wr],[Covr12, Vr2, Covr2Wr], [Covr1Wr, Covr2Wr, 1]])
    
    # The variance of Wr is 1
    sample_r = multivariate_normal(meanr, covr, 1)
    Ir1 = sample_r[0,0]
    Ir2 = sample_r[0,1]
    Wr = sample_r[0,2]
    Ir = e1 * array([[-1J/2 * exp(1J*phi) * Ir1, Wr - Ir2], [exp(2*1J*phi)*Ir2,1J/2* exp(1J*phi) * Ir1]])

    # Deterministic contribution given by relaxation

    # Integral of sin**2(theta/2)
    det1 = (theta -np.sin(theta))/(2*theta)

    #Integral of sin(theta)
    det2 = (1 - np.cos(theta)) / theta

    #Integral of cos**2(theta/2)
    det3 = (theta +np.sin(theta))/(2*theta)

    deterministic = -e1**2/2 * array([[det1, 1J/2*exp(-1J*phi)*det2], [-1J/2*exp(1J*phi)*det2, det3]])

    # Variances and covariances for relaxation Itô processes depending on Z(t)

    # Integral of cos(theta)**2
    Vp_1 = (2*theta + np.sin(2*theta))/(4*theta)

    #Integral of sin(theta)**2
    Vp_2 = (2*theta - np.sin(2*theta))/(4*theta)

    #Integral of sin(theta)*cos(theta)
    Covp_12 = (np.sin(theta))**2/(2*theta)

    meanp = array([0,0])
    covp = array([[Vp_1, Covp_12], [Covp_12, Vp_2]])

    sample_p = multivariate_normal(meanp, covp, 1)
    Ip1 = sample_p[0,0]
    Ip2 = sample_p[0,1]
    Ip = ep * array([[Ip1, -1J * exp(-1J*phi) * Ip2], [1J * exp(1J*phi) * Ip2, -Ip1]])

    result = U @ expm(deterministic) @ expm(1J * Idx + 1J * Idy + 1J * Idz + 1J * Ir + 1J * Ip)
    return result 


def X(phi: float, p: float, T1: float, T2: float) -> np.array:
    """
    NOISY QUANTUM GATE FOR X SINGLE-QUBIT GATES OF IBM's DEVICES (2 order approximated solution, non-unitary matrix)
    
    This function implements the X single-qubit noisy quantum gate with depolarizing and 
    relaxation errors during the unitary evolution.
    
    Args: 
        phi: phase of the drive defining axis of rotation on the Bloch sphere
        p: single-qubit depolarizing error probability
        T1: qubit's amplitude damping time in ns
        T2: qubit's dephasing time in ns
        
    Returns:
          X single-qubit noisy quantum gate
    """
    result = Noise_Gate(pi,phi,p,T1,T2)
    
    return result


def SX(phi: float, p: float, T1: float, T2: float) -> np.array:
    """
    NOISY QUANTUM GATE FOR SX SINGLE-QUBIT GATES OF IBM's DEVICES (2 order approximated solution, non-unitary matrix)
    
    This function implements the SX single-qubit noisy quantum gate with depolarizing and 
    relaxation errors during the unitary evolution.
    
    Args: 
        phi: phase of the drive defining axis of rotation on the Bloch sphere
        p: single-qubit depolarizing error probability
        T1: qubit's amplitude damping time in ns
        T2: qubit's dephasing time in ns
        
    Returns:
          SX single-qubit noisy quantum gate
    """
    result = Noise_Gate(pi/2, phi, p, T1, T2)
    
    return result


def CR(theta: float, phi: float, t_cr: float, p_cr: float, T1_ctr: float, T2_ctr: float, T1_trg: float, T2_trg: float) -> np.array:
    """
    NOISY QUANTUM GATE FOR CROSS RESONANCE (CR) TWO-QUBIT GATE OF IBM's DEVICES (2 order approximated solution, non-unitary matrix)
    
    This function implements the CR two-qubit noisy quantum gate with depolarizing and 
    relaxation errors on both qubits during the unitary evolution.
    
    Args:
        theta: angle of rotation on the Bloch sphere
        phi: phase of the drive defining axis of rotation on the Bloch sphere
        t_cr: CR gate time in ns
        p_cr: CR depolarizing error probability
        T1_ctr: control qubit's amplitude damping time in ns
        T2_ctr: control qubit's dephasing time in ns
        T1_trg: target qubit's amplitude damping time in ns
        T2_trg: target qubit's dephasing time in ns
        
    Returns:
          CR two-qubit noisy quantum gate
    """
    tg = 35 * 10**(-9)
    
    omega = theta
    a = t_cr / tg
    ed_cr = sqrt(p_cr/(4*a)) 
   
    if T1_ctr == 0:
        e1_ctr = 0
    else:
        e1_ctr = sqrt(tg/T1_ctr)
        
    if T2_ctr == 0:
        ep_ctr = 0
    else:
        e2_ctr = sqrt(tg/T2_ctr)
    
        ep_ctr = sqrt((1/2) * (e2_ctr**2 - e1_ctr**2/2))
    
    if T1_trg == 0:
        e1_trg = 0
    else:
        e1_trg = sqrt(tg/T1_trg)
        
    if T2_trg == 0:
        ep_trg = 0
    else:
        e2_trg = sqrt(tg/T2_trg)
    
        ep_trg = sqrt((1/2) * (e2_trg**2 - e1_trg**2/2))
    
    U = array(
        [[cos(theta/2),-1J*sin(theta/2) * exp(-1J * phi),0,0],
        [-1J*sin(theta/2) * exp(1J * phi),cos(theta/2),0,0],
        [0,0,cos(theta/2),1J*sin(theta/2) * exp(-1J * phi)],
        [0,0,1J*sin(theta/2) * exp(1J * phi),cos(theta/2)]]
    )

    #RELAXATION CONTRIBUTIONS

    #Variances and covariances for depolarization Itô processes depending on [tensor(sigma_min,ID)](t)

    Vr_ctr_1 = (2*a*omega + a*sin(2*omega))/(4*omega)
    
    Vr_ctr_2 = 1/4*(2*a - (a*sin(2*omega))/omega)
    
    Covr_ctr_12 = (a*(sin(omega)**2))/(2*omega)
    
    meanr_ctr = [0, 0]
    covr_ctr = [[Vr_ctr_1,Covr_ctr_12],[Covr_ctr_12,Vr_ctr_2]]       
    
    sample_r_ctr = multivariate_normal(meanr_ctr, covr_ctr,1)
    Ir_ctr_1 = sample_r_ctr[0,0]
    Ir_ctr_2 = sample_r_ctr[0,1]
    Ir_ctr = e1_ctr * array(
        [[0, 0, Ir_ctr_1, 1J*Ir_ctr_2 * exp(-1J * phi)],
         [0, 0, 1J*Ir_ctr_2 * exp(1J * phi), Ir_ctr_1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]])

    # Variances and covariances for depolarization Itô processes depending on [tensor(ID,sigma_min)](t)

    # Integral of sin**2(theta)
    Vr_trg_1 = 1/4*(2*a - (a*sin(2*omega))/(omega))
    # Integral of sin**4(theta/2)

    Vr_trg_2 = (a/(16*omega))*(6*omega-8*np.sin(omega)+np.sin(2*omega))
    # Integral of sin(theta) sin**2(theta/2)

    Covr_trg_12 = (a/omega)*(np.sin(omega/2))**4
    # Integral of sin(theta)

    Covr_trg_1Wr_trg = (a/omega)*(1-np.cos(omega))

    # Integral of sin**2(theta)
    Covr_trg_2Wr_trg = (a/(2*omega))*(omega - np.sin(omega))

    # The variance of W2r is a
    meanr_trg = [0, 0, 0]
    covr_trg = [[Vr_trg_1,Covr_trg_12,Covr_trg_1Wr_trg],
                [Covr_trg_12,Vr_trg_2,Covr_trg_2Wr_trg],
                [Covr_trg_1Wr_trg,Covr_trg_2Wr_trg,a]]
    
    sample_r_trg = multivariate_normal(meanr_trg, covr_trg,1)
    Ir_trg_1 = sample_r_trg[0,0]
    Ir_trg_2 = sample_r_trg[0,1]
    Wr_trg = sample_r_trg[0,2]
    Ir_trg = e1_trg * array(
        [[-1J*(1/2)*Ir_trg_1*exp(1J*phi), Wr_trg-Ir_trg_2, 0, 0],
         [Ir_trg_2*exp(2*1J*phi), 1J*(1/2)*Ir_trg_1*exp(1J*phi), 0, 0],
         [0, 0, 1J*(1/2)*Ir_trg_1*exp(1J*phi),Wr_trg-Ir_trg_2],
         [0, 0, Ir_trg_2*exp(2*1J*phi), -1J*(1/2)*Ir_trg_1*exp(1J*phi)]]
    )

    # Variances and covariances for depolarization Itô processes depending on [tensor(Z,ID)](t)

    Wp_ctr = normal(0, sqrt(a))
    Ip_ctr = ep_ctr * array(
        [[Wp_ctr, 0, 0, 0],
        [0, Wp_ctr, 0, 0],
        [0, 0, -Wp_ctr, 0],
        [0, 0, 0, -Wp_ctr]]
    )

    # Variances and covariances for depolarization Itô processes depending on [tensor(ID,Z)](t)

    Vp_trg_1 = (2*a*omega + a*sin(2*omega))/(4*omega)
    
    Vp_trg_2 = 1/4*(2*a - (a*sin(2*omega))/(omega))
    
    Covp_trg_12 = (a*(sin(omega)**2))/(2*omega)
    
    meanp_trg = [0, 0]
    covp_trg = [[Vp_trg_1, Covp_trg_12], [Covp_trg_12, Vp_trg_2]]
    
    sample_p_trg = multivariate_normal(meanp_trg, covp_trg, 1)
    Ip_trg_1 = sample_p_trg[0,0]
    Ip_trg_2 = sample_p_trg[0,1]
    Ip_trg = ep_trg * array(
        [[Ip_trg_1, -1J*Ip_trg_2*exp(-1J*phi), 0, 0],
         [1J*Ip_trg_2*exp(1J*phi), -Ip_trg_1, 0, 0],
         [0, 0, Ip_trg_1, 1J*Ip_trg_2*exp(-1J*phi)],
         [0, 0, -1J*Ip_trg_2*exp(1J*phi), -Ip_trg_1]]
    )
    
    #Deterministic contribution given by relaxation
    
    det1 = (a*omega-a*sin(omega))/(2*omega)
   
    det2 = (a/omega)*(1-cos(omega))
    
    det3 = a/(2*omega)*(omega+sin(omega))
    
    deterministic_r_ctr = -e1_ctr**2/2 * array([[0,0,0,0],[0,0,0,0],[0,0,a,0],[0,0,0,a]])
    deterministic_r_trg = -e1_trg**2/2 * array(
        [[det1,1J*(1/2)*det2*exp(-1J*phi),0,0],
        [-1J*(1/2)*det2*exp(1J*phi),det3,0,0],
        [0,0,det1,-1J*(1/2)*det2*exp(-1J*phi)],[0,0,1J*(1/2)*det2*exp(1J*phi),det3]]
    )
    
    #DEPOLARIZATION CONTRIBUTIONS

    #Variances and covariances for depolarization Itô processes depending on [tensor(X,ID)](t)

    Vdx_ctr_1 = (2*a*omega + a*sin(2*omega))/(4*omega)
    
    Vdx_ctr_2 = 1/4*(2*a - (a*sin(2*omega))/(omega))
    
    Covdx_ctr_12 = (a*(sin(omega)**2))/(2*omega)
    
    meandx_ctr = [0, 0]
    covdx_ctr = [[Vdx_ctr_1, Covdx_ctr_12], [Covdx_ctr_12, Vdx_ctr_2]]
    
    sample_dx_ctr = multivariate_normal(meandx_ctr, covdx_ctr, 1)
    Idx_ctr_1 = sample_dx_ctr[0,0]
    Idx_ctr_2 = sample_dx_ctr[0,1]
    Idx_ctr = ed_cr * array(
        [[0, 0, Idx_ctr_1, 1J*Idx_ctr_2 * exp(-1J * phi)],
         [0, 0, 1J*Idx_ctr_2 * exp(1J * phi), Idx_ctr_1],
         [Idx_ctr_1, -1J*Idx_ctr_2 * exp(-1J * phi), 0, 0],
         [-1J*Idx_ctr_2 * exp(1J * phi), Idx_ctr_1, 0, 0]]
    )

    #Variances and covariances for depolarization Itô processes depending on [tensor(Y,ID)](t)

    Vdy_ctr_1 = (2*a*omega + a*sin(2*omega))/(4*omega)
    
    Vdy_ctr_2 = 1/4*(2*a - (a*sin(2*omega))/(omega))
    
    Covdy_ctr_12 = (a*(sin(omega)**2))/(2*omega)
    
    meandy_ctr = [0, 0]
    covdy_ctr = [[Vdy_ctr_1, Covdy_ctr_12], [Covdy_ctr_12, Vdy_ctr_2]]
    
    sample_dy_ctr = multivariate_normal(meandy_ctr, covdy_ctr, 1)
    Idy_ctr_1 = sample_dy_ctr[0,0]
    Idy_ctr_2 = sample_dy_ctr[0,1]
    Idy_ctr = ed_cr * array(
        [[0, 0, -1J*Idy_ctr_1, Idy_ctr_2 * exp(-1J * phi)],
         [0, 0, Idy_ctr_2 * exp(1J * phi), -1J*Idy_ctr_1],
         [1J*Idy_ctr_1, Idy_ctr_2 * exp(-1J * phi), 0, 0],
         [Idy_ctr_2 * exp(1J * phi), 1J*Idy_ctr_1, 0, 0]]
    )

    #Variances and covariances for depolarization Itô processes depending on [tensor(Z,ID)](t)

    Wdz_ctr = normal(0, sqrt(a))
    Idz_ctr = ed_cr * array([[Wdz_ctr,0,0,0],[0,Wdz_ctr,0,0],[0,0,-Wdz_ctr,0],[0,0,0,-Wdz_ctr]])

    #Variances and covariances for depolarization Itô processes depending on [tensor(ID,X)](t)

    #Integral of sin**2(theta)
    Vdx_trg_1 = 1/4*(2*a - (a*sin(2*omega))/(omega))

    #Integral of sin**4(theta/2)
    Vdx_trg_2 = (a/(16*omega))*(6*omega-8*np.sin(omega)+np.sin(2*omega))

    #Integral of sin(theta) sin**2(theta/2)

    Covdx_trg_12 = (a/omega)*(np.sin(omega/2))**4

    #Integral of sin(theta)
    Covdx_trg_1Wdx = (a/omega)*(1-np.cos(omega))

    #Integral of sin**2(theta)
    Covdx_trg_2Wdx = (a/(2*omega))*(omega - np.sin(omega))

    meandx_trg = array([0, 0, 0])
    covdx_trg = array(
        [[Vdx_trg_1, Covdx_trg_12, Covdx_trg_1Wdx],
        [Covdx_trg_12, Vdx_trg_2, Covdx_trg_2Wdx],
        [Covdx_trg_1Wdx, Covdx_trg_2Wdx, a]]
    )
    
    # the variance of Wdx is a
    sample_dx_trg = multivariate_normal(meandx_trg, covdx_trg,1)
    Idx_trg_1 = sample_dx_trg[0,0]
    Idx_trg_2 = sample_dx_trg[0,1]
    Wdx_trg = sample_dx_trg[0,2]
    Idx_trg = ed_cr * array(
        [[Idx_trg_1 * np.sin(phi), Wdx_trg + (exp(-2*1J*phi)-1)*Idx_trg_2, 0, 0],
         [Wdx_trg + (exp(2*1J*phi)-1)*Idx_trg_2, -Idx_trg_1*np.sin(phi), 0, 0],
         [0,  0, -Idx_trg_1 * np.sin(phi), Wdx_trg + (exp(-2*1J*phi)-1)*Idx_trg_2],
         [0, 0, Wdx_trg + (exp(2*1J*phi)-1)*Idx_trg_2, Idx_trg_1 * np.sin(phi)]]
    )

    #Variances and covariances for depolarization Itô processes depending on [tensor(ID,Y)](t)
    
    #Integral of sin**2(theta)
   
    Vdy_trg_1 = 1/4*(2*a - (a*sin(2*omega))/(omega))

    #Integral of sin**4(theta/2)
    Vdy_trg_2 = (a/(16*omega))*(6*omega-8*np.sin(omega)+np.sin(2*omega))
    
    #Integral of sin(theta) sin**2(theta/2)
    Covdy_trg_12 = (a/omega)*(np.sin(omega/2))**4
    
    #Integral of sin(theta)
    Covdy_trg_1Wdy = (a/omega)*(1-np.cos(omega))
    
    #Integral of sin**2(theta)
    Covdy_trg_2Wdy = (a/(2*omega))*(omega - np.sin(omega))
    
    meandy_trg = array([0, 0, 0])
    covdy_trg = array(
        [[Vdy_trg_1, Covdy_trg_12, Covdy_trg_1Wdy],
         [Covdy_trg_12, Vdy_trg_2, Covdy_trg_2Wdy],
         [Covdy_trg_1Wdy, Covdy_trg_2Wdy, a]]
    )
    
    #the variance of Wdy is a

    sample_dy_trg = multivariate_normal(meandy_trg, covdy_trg,1)
    Idy_trg_1 = sample_dy_trg[0,0]
    Idy_trg_2 = sample_dy_trg[0,1]
    Wdy_trg = sample_dy_trg[0,2]
    Idy_trg = ed_cr * array(
        [[-Idy_trg_1*np.cos(phi), -1J*Wdy_trg + 1J * (exp(-2*1J*phi)+1)*Idy_trg_2, 0, 0],
         [1J*Wdy_trg - 1J * (exp(2*1J*phi)+1)*Idy_trg_2, Idy_trg_1*np.cos(phi), 0, 0],
         [0, 0, Idy_trg_1*np.cos(phi), -1J*Wdy_trg + 1J * (exp(-2*1J*phi)+1)*Idy_trg_2],
         [0, 0, 1J*Wdy_trg - 1J * (exp(2*1J*phi)+1)*Idy_trg_2, -Idy_trg_1*np.cos(phi)]]
    )

    #Variances and covariances for depolarization Itô processes depending on [tensor(ID,Z)](t)
    
    Vdz_trg_1 = (2*a*omega + a*sin(2*omega))/(4*omega)
    
    Vdz_trg_2 = 1/4*(2*a - (a*sin(2*omega))/(omega))
    
    Covdz_trg_12 = (a*(sin(omega)**2))/(2*omega)
    
    meandz_trg = [0, 0]
    covdz_trg = [[Vdz_trg_1, Covdz_trg_12], [Covdz_trg_12, Vdz_trg_2]]
    
    sample_dz_trg = multivariate_normal(meandz_trg, covdz_trg,1)
    Idz_trg_1 = sample_dz_trg[0,0]
    Idz_trg_2 = sample_dz_trg[0,1]
    Idz_trg = ed_cr * array(
        [[Idz_trg_1, -1J*Idz_trg_2*exp(-1J*phi), 0, 0],
         [1J*Idz_trg_2*exp(1J*phi), -Idz_trg_1, 0, 0],
         [0, 0, Idz_trg_1, 1J*Idz_trg_2*exp(-1J*phi)],
         [0, 0, -1J*Idz_trg_2*exp(1J*phi), -Idz_trg_1]]
    )

    result = U @ expm(deterministic_r_ctr + deterministic_r_trg)\
               @ expm(1J * Ir_ctr + 1J * Ir_trg + 1J * Ip_ctr + 1J * Ip_trg + 1J * Idx_ctr + 1J * Idy_ctr + 1J * Idz_ctr + 1J * Idx_trg + 1J * Idy_trg + 1J * Idz_trg)
    return result


def CNOT(phi_ctr: float, phi_trg: float, t_cnot: float, p_cnot: float, p_single_ctr: float, p_single_trg: float,
         T1_ctr: float, T2_ctr: float, T1_trg: float, T2_trg: float) -> np.array:
    """
    NOISY QUANTUM GATE FOR CNOT TWO-QUBIT GATE OF IBM's DEVICES (2 order approximated solution, non-unitary matrix)
    
    This function implements the CNOT two-qubit noisy quantum gate with depolarizing and 
    relaxation errors on both qubits during the unitary evolution.
    
    Args:
        phi_ctr: control qubit phase of the drive defining axis of rotation on the Bloch sphere
        phi_trg: target qubit phase of the drive defining axis of rotation on the Bloch sphere
        t_cnot: CNOT gate time in ns
        p_cnot: CNOT depolarizing error probability
        p_single_ctr: control qubit depolarizing error probability
        p_single_trg: target qubit depolarizing error probability
        T1_ctr: control qubit's amplitude damping time in ns
        T2_ctr: control qubit's dephasing time in ns
        T1_trg: target qubit's amplitude damping time in ns
        T2_trg: target qubit's dephasing time in ns
        
    Returns:
          CNOT two-qubit noisy quantum gate
    """
    tg = 35*10**(-9)
    
    t_cr = t_cnot/2-tg
    
    p_cr = (4/3) * (1 - sqrt(sqrt((1 - (3/4) * p_cnot)**2 / ((1-(3/4)*p_single_ctr)**2 * (1-(3/4)*p_single_trg)))))
    
    Y_Rz = Noise_Gate(-pi, -phi_ctr + pi/2 + pi/2, p_single_ctr, T1_ctr, T2_ctr)
    
    result = CR(-pi / 4, -phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg) \
             @ kron(X(-phi_ctr + pi / 2, p_single_ctr, T1_ctr, T2_ctr), relaxation(tg, T1_trg, T2_trg)) \
             @ CR(pi / 4, -phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg) \
             @ kron(Y_Rz, SX(-phi_trg, p_single_trg, T1_trg, T2_trg))
    
    return result


def CNOT_inv(phi_ctr: float, phi_trg: float, t_cnot: float, p_cnot: float, p_single_ctr: float,
             p_single_trg: float, T1_ctr: float, T2_ctr: float, T1_trg: float, T2_trg: float) -> np.array:
    """
    NOISY QUANTUM GATE FOR REVERSE CNOT TWO-QUBIT GATE OF IBM's DEVICES (2 order approximated solution, non-unitary matrix)
      
    This function implements the reverse CNOT two-qubit noisy quantum gate with depolarizing and 
    relaxation errors on both qubits during the unitary evolution.
    
    Args:
        phi_ctr: control qubit phase of the drive defining axis of rotation on the Bloch sphere
        phi_trg: target qubit phase of the drive defining axis of rotation on the Bloch sphere
        t_cnot: reverse CNOT gate time in ns
        p_cnot: reverse CNOT depolarizing error probability
        p_single_ctr: control qubit depolarizing error probability
        p_single_trg: target qubit depolarizing error probability
        T1_ctr: control qubit's amplitude damping time in ns
        T2_ctr: control qubit's dephasing time in ns
        T1_trg: target qubit's amplitude damping time in ns
        T2_trg: target qubit's dephasing time in ns
        
    Returns:
           reverse CNOT two-qubit noisy quantum gate
    """
    tg = 35*10**(-9)
    
    t_cr = (t_cnot-3*tg)/2
    
    p_cr = (4/3) * (1 - sqrt(sqrt((1 - (3/4) * p_cnot)**2 / ((1-(3/4)*p_single_ctr)**2 * (1-(3/4)*p_single_trg)**3))))
    
    Ry = Noise_Gate(-pi/2, -phi_trg-pi/2+pi/2, p_single_trg, T1_trg, T2_trg)
    
    Y_Z = Noise_Gate(pi/2, -phi_ctr-pi+pi/2, p_single_ctr, T1_ctr, T2_ctr)
    
    result = kron(Ry, SX(-phi_ctr - pi - pi / 2, p_single_ctr, T1_ctr, T2_ctr)) \
             @ CR(-pi / 4, -phi_ctr - pi, t_cr, p_cr, T1_trg, T2_trg, T1_ctr, T2_ctr) \
             @ kron(X(-phi_trg - pi / 2, p_single_trg, T1_trg, T2_trg), relaxation(tg, T1_ctr, T2_ctr)) \
             @ CR(pi / 4, -phi_ctr - pi, t_cr, p_cr, T1_trg, T2_trg, T1_ctr, T2_ctr) \
             @ kron(SX(-phi_trg - pi / 2, p_single_ctr, T1_ctr, T2_ctr), Y_Z)
    
    return result


class LegacyGates(object):
    """ Collection of the legacy gates.
    """

    relaxation = staticmethod(relaxation)
    bitflip = staticmethod(bitflip)
    depolarizing = staticmethod(depolarizing)
    X = staticmethod(X)
    SX = staticmethod(SX)
    CNOT = staticmethod(CNOT)
    CNOT_inv = staticmethod(CNOT_inv)
