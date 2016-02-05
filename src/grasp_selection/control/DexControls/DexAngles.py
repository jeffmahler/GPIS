from numpy import rad2deg
class DexAngles:

    def __init__(self, phi, psi, gamma):
        self.phi = phi
        self.phi_deg = rad2deg(phi)
        self.psi = psi
        self.psi_deg = rad2deg(psi)
        self.gamma = gamma
        self.gamma_deg = rad2deg(gamma)
        
    @property
    def roll(self):
        return self.gamma
        
    @property
    def pitch(self):
        return self.psi
        
    @property
    def yaw(self):
        return self.phi    
        
    @property
    def roll_deg(self):
        return self.gamma_deg
        
    @property
    def pitch_deg(self):
        return self.psi_deg
        
    @property
    def yaw_deg(self):
        return self.phi_deg
        
    
    