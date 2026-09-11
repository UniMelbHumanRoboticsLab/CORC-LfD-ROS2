import numpy as np
np.set_printoptions(suppress=True,precision=10) # suppress scientific notation
from tqdm import tqdm
realmin  = np.finfo(np.float64).tiny
realmax  = np.finfo(np.float64).max

import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from tpgmm_objects import Sample,TaskParams

"""
Class for TPGMM
"""
""" get max norm diff between 2 n dimensional arrays"""
def get_max_norm_diff(arr1,arr2):
    diff = np.abs(arr1 - arr2)
    max_index = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
    max_norm_diff = diff[max_index]/(np.abs(arr2[max_index])+realmin)
    return max_norm_diff

class TPGMM:
    def __init__(self, num_of_gauss, num_of_frames, num_of_dim,priors,kP,kV,diagRegFact,version="fast",init_type='kmeans++'):
        self.model_id = ""
        self.bic = ""
        self.bic_time = 0
        self.training_time = 0
        self.training_status = 0
        self.num_of_gauss = num_of_gauss
        self.num_of_frames = num_of_frames
        self.num_of_dim = num_of_dim
        self.converged_LL = 0
        self.priors = priors
        self.kP = kP
        self.kV = kV
        self.diagRegFact = diagRegFact
        self.version = version
        self.init_type = init_type
        print(f"Initialized TPGMM with {num_of_gauss} Gaussians, {num_of_frames} frames, {num_of_dim} dimensions, version: {version}, init_type: {init_type}")
        self.conditioned=False

    """
    initialize the GMM parameters for each Gaussian in each frame
    """
    def init_gmm (self,tp_data):
        # time window init
        if self.init_type == 'time':
            tp_data = np.reshape(tp_data.T,(tp_data.shape[1]*tp_data.shape[2],tp_data.shape[0]))

            Mu = np.zeros(shape=(self.num_of_frames*self.num_of_dim, self.num_of_gauss))
            Sigma = np.zeros(shape=(self.num_of_frames*self.num_of_dim, self.num_of_frames*self.num_of_dim, self.num_of_gauss))
            sampledTime = tp_data[0,:]
            TimingSep = np.linspace(np.min(sampledTime),np.max(sampledTime),self.num_of_gauss+1)
            # initialize the mean and covariance matrices within each time window
            for i in range(self.num_of_gauss):
                window = np.where(np.logical_and(sampledTime>=TimingSep[i],sampledTime<TimingSep[i+1]) == True)[0]
                Mu[:,i] = np.mean(tp_data[:,window],1)
                Sigma[:,:,i] = np.cov(tp_data[:,window]) + np.eye(tp_data.shape[0])*self.diagRegFact
                self.priors[i] = window.shape[0]
            self.priors = self.priors / sum(self.priors); 
            # reshape the mean and sigma array (num_of_dim,num_of_frames,num_of_gauss)
            self.Mu = np.zeros((self.num_of_dim,self.num_of_frames,self.num_of_gauss))
            self.Sigma = np.zeros(shape=(self.num_of_dim,self.num_of_dim,self.num_of_frames,self.num_of_gauss))
            for i in range(self.num_of_gauss):
                self.Mu[:,:,i] = np.reshape(Mu[:,i],(self.num_of_frames,self.num_of_dim)).T
                for j in range(self.num_of_frames):
                    sigmaCurGaussCurFrame = Sigma[j*self.num_of_dim:(j+1)*self.num_of_dim,j*self.num_of_dim:(j+1)*self.num_of_dim,i]
                    self.Sigma[:,:,j,i] = sigmaCurGaussCurFrame
        # Kmeans++ init
        else:
            from gmr.gmm import kmeansplusplus_initialization
            
            # start with equal priors for each gaussian
            self.priors = np.random.dirichlet(np.ones(self.num_of_gauss))
            # self.priors = self.priors / self.num_of_gauss
            self.Mu = np.zeros((self.num_of_dim,self.num_of_frames,self.num_of_gauss))
            self.Sigma = np.zeros(shape=(self.num_of_dim,self.num_of_dim,self.num_of_frames,self.num_of_gauss))
            
            # save a copy of the initial parameters
            self.Mu_init = self.Mu.copy()
            self.Sigma_init = self.Sigma.copy()
            self.priors_init = self.priors.copy()
            
            
            for i in range(self.num_of_frames):
                dataCurFrame = tp_data[:,:,i]
                # initialize the mean for current frame
                MuCurFrame = kmeansplusplus_initialization(
                    dataCurFrame, self.num_of_gauss
                )

                average_distances = np.std(dataCurFrame, axis=0)
                SigmaCurFrame = np.empty((self.num_of_gauss, self.num_of_dim, self.num_of_dim))
                SigmaCurFrame[:]  = (np.eye(self.num_of_dim) *
                                          (average_distances / self.num_of_gauss) ** 2)
                
                # # initialize the covariance for current frame
                # from gmr.gmm import covariance_initialization
                # SigmaCurFrame = covariance_initialization(
                #     dataCurFrame,self.num_of_gauss)
                
                # n_components, n_features = means.shape
                # covariances = xp.empty(
                #     (n_components, n_features, n_features), device=device_, dtype=X.dtype
                # )
                # for k in range(n_components):
                #     diff = X - means[k, :]
                #     covariances[k, :, :] = ((resp[:, k] * diff.T) @ diff) / nk[k]
                #     _add_to_diagonal(covariances[k, :, :], reg_covar, xp)
                # return covariances

                for j in range(self.num_of_gauss):
                    # print(j)
                    self.Mu[:,i,j] = MuCurFrame[j]
                    self.Sigma[:,:,i,j] = SigmaCurFrame[j] + np.eye(self.num_of_dim)*self.diagRegFact
            
            # save a copy of the initial parameters
            self.Mu_init = self.Mu.copy()
            self.Sigma_init = self.Sigma.copy()
            self.priors_init = self.priors.copy()
            print(f"Init Priors:\t{self.priors}\tSum:{np.sum(self.priors)}\n")

    """
    perform EM to get the most optimal mean and covariance for each Gaussian in each frame
    """
    def computeGamma(self,tp_data):
        # ============ Vectorized version ============
        # tp_data: (nbData, num_of_dim, num_of_frames)
        # self.Mu: (num_of_dim, num_of_frames, num_of_gauss)
        # self.Sigma: (num_of_dim, num_of_dim, num_of_frames, num_of_gauss)

        # Expand tp_data for broadcasting: (nbData, num_of_dim, num_of_frames, 1)
        data_expanded = tp_data[:, :, :, np.newaxis]

        # Expand Mu for broadcasting: (1, num_of_dim, num_of_frames, num_of_gauss)
        Mu_expanded = self.Mu[np.newaxis, :, :, :]

        # Compute difference: (nbData, num_of_dim, num_of_frames, num_of_gauss)
        diff = data_expanded - Mu_expanded

        # Transpose Sigma for batch inversion: (num_of_frames, num_of_gauss, num_of_dim, num_of_dim)
        Sigma_transposed = np.transpose(self.Sigma, (2, 3, 0, 1))
        inv_Sigma = np.linalg.inv(Sigma_transposed)

        # Transpose diff to (num_of_frames, num_of_gauss, num_of_dim, nbData)
        diff_transposed = np.transpose(diff, (2, 3, 1, 0))

        # Compute quadratic form: diff.T @ inv_Sigma @ diff for each (frame, gauss, data)
        # inv_Sigma @ diff: (f, g, d, n)
        inv_sigma_diff = np.einsum('fgij,fgjn->fgin', inv_Sigma, diff_transposed)
        # Sum over dim: (f, g, n)
        quad = np.einsum('fgin,fgin->fgn', inv_sigma_diff, diff_transposed)

        # Compute determinants: (num_of_frames, num_of_gauss)
        dets = np.linalg.det(Sigma_transposed)

        # Normalization constant: (num_of_frames, num_of_gauss)
        norm = np.sqrt((2 * np.pi) ** self.num_of_dim * np.abs(dets))

        # Compute PDF: (num_of_frames, num_of_gauss, nbData)
        pdf = np.exp(-0.5 * quad) / norm[:, :, np.newaxis]

        # GAMMA02: (num_of_gauss, num_of_frames, nbData)
        GAMMA0 = np.transpose(pdf, (1, 0, 2))

        # Lik2: product over frames, multiplied by priors
        Lik_unclipped = np.prod(GAMMA0, axis=1) * self.priors[:, np.newaxis]
        Lik = np.clip(Lik_unclipped, realmin, realmax) # to prevent numerical issues

        # GAMMA2: normalized responsibilities
        GAMMA = Lik / np.sum(Lik, axis=0)
        
        return Lik, GAMMA, GAMMA0
    def fit_em(self,nbMinSteps,nbMaxSteps,maxDiffLL,updateComp,tp_data):
        """
        Fit the data into the TPGMM model

        nbMinSteps:min number of allowed iterations
        nbMaxSteps:max number of allowed iterations
        maxDiffL:Likelihood increase threshold to stop the algorithm
        updateComp:flag to update prior,sigma and mu
        """    
        nbData = tp_data.shape[0]
        LL = []
        pbar = tqdm(range(nbMaxSteps), ncols=120)

        for iter in pbar:
            # Fast EM
            if self.version=="fast" or self.version=="compare":
                # E-step (vectorized)
                L, GAMMA, GAMMA0 = self.computeGamma(tp_data)
                GAMMA_SUM = GAMMA / np.expand_dims(np.sum(GAMMA,1),axis=-1)
                self.Pix = GAMMA_SUM

                # M-step (vectorized)
                priors = self.priors
                Mu = self.Mu
                Sigma = self.Sigma
                # Update Priors: (num_of_gauss,)
                if updateComp[0]:
                    priors = np.sum(GAMMA, axis=1) / nbData
                # Update Mu: (num_of_dim, num_of_frames, num_of_gauss)
                # tp_data: (nbData, num_of_dim, num_of_frames), GAMMA2: (num_of_gauss, nbData)
                if updateComp[1]:
                    Mu = np.einsum('ndf,gn->dfg', tp_data, GAMMA_SUM)
                else:
                    Mu = self.Mu.copy()
                # Update Sigma: (num_of_dim, num_of_dim, num_of_frames, num_of_gauss)
                if updateComp[2]:
                    # Compute difference using the UPDATED Mu: (nbData, num_of_dim, num_of_frames, num_of_gauss)
                    diff = tp_data[:, :, :, np.newaxis] - Mu[np.newaxis, :, :, :]
                    # Weighted outer product summed over data points
                    # diff shape: (n, d, f, g), GAMMA2 shape: (g, n)
                    # Output Sigma shape: (d, d, f, g)
                    Sigma = np.einsum('gn,nafg,nbfg->abfg', GAMMA_SUM, diff, diff) + np.eye(self.num_of_dim)[:, :, np.newaxis, np.newaxis] * self.diagRegFact # add regularization to prevent singular matrix
                # compute Average Log-likelihood to estimate convergence
                curLL = np.sum(np.log(np.sum(L,0))) / L.shape[1]

            # assign updated parameters
            if self.version=="fast":
                self.priors = priors
                self.Mu = Mu
                self.Sigma = Sigma

            LL.append(curLL)
            diff = np.abs((LL[iter]-LL[iter-1]))
            pbar.set_postfix(iter=iter,cur_ll=curLL,loglike_diff=diff)

            if iter>nbMinSteps:
                if diff<maxDiffLL or  iter==nbMaxSteps:
                    break

        print(f"Likelihood {iter}: {LL[iter]:.4f} \t Tol: {diff:.4f}")
        self.converged_LL = LL
        print(f"Conv. Priors:\t{self.priors}\tSum:{np.sum(self.priors)}")
        print(f'EM converged ({"Success" if iter!=nbMaxSteps else "Fail"}) after {iter+1} iterations.')
        
        fail = False
        if LL[iter] < 0 or iter==nbMaxSteps:
            fail = True
        return np.array([LL]).T,fail
    
    """
    reproduce optimal trajectory by conditioning GMM with task parameters then GMR (with slow and fast implementation)
    """
    def dynSysControl(self,DataIn,expected_data,sampleParam:TaskParams,last_input_index,last_output_index,new_dt):
        if len(DataIn.shape) == 1: # to standardize for multi-input cases
            DataIn = np.expand_dims(DataIn,axis=-1)
        nbData = DataIn.shape[0]
        nbOut = last_output_index-last_input_index
        X = sampleParam.params.b[0,last_input_index+1:last_output_index+1]
        dX = np.zeros((nbOut,1))
        L = np.hstack((np.eye(nbOut)*self.kP, np.eye(nbOut)*self.kV))
        Input_reconOutput = np.zeros((self.num_of_dim,nbData))
        for t in range(nbData):
            np.vstack((X-np.expand_dims(expected_data[:,t],axis=-1),dX))
            ddx = -np.matmul(L,np.vstack((X-np.expand_dims(expected_data[:,t],axis=-1),dX)))
            # update the velocity with the corrected acceleration
            dX = dX + ddx * new_dt
            X = X + dX * new_dt
            if len(DataIn[t,:].shape) == 1: # to standardize for multi-input cases
                DataInTmp = np.expand_dims(DataIn[t,:],axis=-1)
            Input_reconOutput[:,t] = np.squeeze(np.vstack((DataInTmp,X)),axis=-1)
        expectedOutput = Input_reconOutput[last_input_index+1:last_output_index+1,:]
        return Input_reconOutput,expectedOutput # return
    def conditionTPGMM(self,sampleParam:TaskParams,last_input_index,last_output_index):
        """
        Recondition the TPGMM using product of linearly transformed Gaussian
        """
        self.nbOut = last_output_index - last_input_index
        self.last_output_index = last_output_index+1
        self.nbIn = last_input_index+1
        
        # Fast version
        # Stack A and b from sampleParam: A shape (f, d, d), b shape (f, d, 1)
        A = np.array([sampleParam.A[j] for j in range(self.num_of_frames)])  # (f, d, d)
        b = np.array([sampleParam.b[j] for j in range(self.num_of_frames)])  # (f, d, 1)
        
        # self.Mu: (d, f, g), self.Sigma: (d, d, f, g)
        
        # Transform Mu for all frames and Gaussians: muCurFrame = A @ Mu + b
        # Mu transposed to (f, d, g) for einsum
        Mu_t = np.transpose(self.Mu, (1, 0, 2))  # (f, d, g)
        # A @ Mu for each frame: (f, d, d) @ (f, d, g) -> (f, d, g)
        muCurFrame = np.einsum('fij,fjg->fig', A, Mu_t) + b  # (f, d, g)
        
        # Transform Sigma for all frames and Gaussians: sigmaCurFrame = A @ Sigma @ A.T
        # Sigma transposed to (f, g, d, d) for batch operations
        Sigma_t = np.transpose(self.Sigma, (2, 3, 0, 1))  # (f, g, d, d)
        # A @ Sigma @ A.T: (f, d, d) @ (f, g, d, d) @ (f, d, d).T -> (f, g, d, d)
        sigmaCurFrame = np.einsum('fij,fgjk,flk->fgil', A, Sigma_t, A)  # (f, g, d, d)
        
        # Invert sigmaCurFrame for all frames and Gaussians
        inv_sigmaCurFrame = np.linalg.inv(sigmaCurFrame)  # (f, g, d, d)
        
        # Sum inverse covariances over frames: sigmaTmp = sum_j inv(sigmaCurFrame[j])
        sigmaTmp = np.sum(inv_sigmaCurFrame, axis=0)  # (g, d, d)
        
        # Compute MuTmp = sum_j inv(sigmaCurFrame[j]) @ muCurFrame[j]
        # muCurFrame is (f, d, g), transpose to (f, g, d) for matrix multiply
        muCurFrame_t = np.transpose(muCurFrame, (0, 2, 1))  # (f, g, d)
        # inv_sigmaCurFrame @ muCurFrame and sum over frames
        MuTmp = np.einsum('fgij,fgj->gi', inv_sigmaCurFrame, muCurFrame_t)  # (g, d)
        
        # Final Sigma = inv(sigmaTmp)
        Sigma = np.linalg.inv(sigmaTmp)  # (g, d, d)
        
        # Final Mu = Sigma @ MuTmp
        Mu = np.einsum('gij,gj->gi', Sigma, MuTmp)  # (g, d)
        
        # Transpose to match original output shapes: Sigma (d, d, g), Mu (d, g)
        tpNewSigma = np.transpose(Sigma, (1, 2, 0))  # (d, d, g)
        tpNewMu = Mu.T  # (d, g) 
        
        # Precompute slices for all Gaussians
        # MuIn: (nbIn, g), MuOut: (nbOut, g)
        MuIn = tpNewMu[0:self.nbIn, :]  # (nbIn, g)
        self.MuOut = tpNewMu[self.nbIn:self.last_output_index, :]  # (nbOut, g)
        # SigmaIn: (nbIn, nbIn, g), SigmaOut: (nbOut, nbOut, g)
        SigmaIn = tpNewSigma[0:self.nbIn, 0:self.nbIn, :]  # (nbIn, nbIn, g)
        SigmaOut = tpNewSigma[self.nbIn:self.last_output_index, self.nbIn:self.last_output_index, :]  # (nbOut, nbOut, g)
        
        # Cross-covariances
        SigmaOutIn = tpNewSigma[self.nbIn:self.last_output_index, 0:self.nbIn, :]  # (nbOut, nbIn, g)
        SigmaInOut = tpNewSigma[0:self.nbIn, self.nbIn:self.last_output_index, :]  # (nbIn, nbOut, g)
        
        # Transpose to have g first for batch operations
        SigmaIn_t = np.transpose(SigmaIn, (2, 0, 1))  # (g, nbIn, nbIn)
        SigmaOut_t = np.transpose(SigmaOut, (2, 0, 1))  # (g, nbOut, nbOut)
        SigmaOutIn_t = np.transpose(SigmaOutIn, (2, 0, 1))  # (g, nbOut, nbIn)
        SigmaInOut_t = np.transpose(SigmaInOut, (2, 0, 1))  # (g, nbIn, nbOut)
        
        # Batch invert SigmaIn
        self.SigmaIn_inv = np.linalg.inv(SigmaIn_t)  # (g, nbIn, nbIn)
        
        # Precompute K = SigmaOutIn @ SigmaIn_inv for all Gaussians
        self.K = np.einsum('goj,gji->goi', SigmaOutIn_t, self.SigmaIn_inv)  # (g, nbOut, nbIn)
        
        # Precompute SigmaTmp_all = SigmaOut - K @ SigmaInOut for all Gaussians (time-independent)
        K_SigmaInOut = np.einsum('goj,gjp->gop', self.K, SigmaInOut_t)  # (g, nbOut, nbOut)
        self.SigmaTmp_all = SigmaOut_t - K_SigmaInOut  # (g, nbOut, nbOut)
        
        # For Gaussian PDF computation
        self.MuIn_t = MuIn.T  # (g, nbIn)
        detSigmaIn = np.linalg.det(SigmaIn_t)  # (g,)
        self.norm = np.sqrt((2 * np.pi) ** self.nbIn * np.abs(detSigmaIn))  # (g,)
        self.conditioned=True
    def GMR_sample(self, input_sample):
        """
        GMR for a single sample
        """
        # ===== Vectorized activation weights (priorsOutput) =====
        diff_in = input_sample - self.MuIn_t  # (g, nbIn)
        
        # Quadratic form: diff_in @ SigmaIn_inv @ diff_in.T for each g
        tmp = np.einsum('gi,gij->gj', diff_in, self.SigmaIn_inv)  # (g, nbIn)
        quad = np.einsum('gi,gi->g', tmp, diff_in)  # (g,)
        
        # Compute PDF and activation weights
        pdf_unclip = np.exp(-0.5 * quad) / self.norm  # (g,)
        weighted_pdf = self.priors * pdf_unclip  # (g,)
        priorsOutput = np.clip(weighted_pdf, realmin, realmax) # prevent numerical issues
        priorsOutput = priorsOutput / (np.sum(priorsOutput))
        
        # ===== Vectorized conditional means (muTmp) =====
        # muTmp = MuOut + K @ diff_in
        # K: (g, nbOut, nbIn), diff_in: (g, nbIn) -> (g, nbOut)
        muTmp_t = np.einsum('goj,gj->go', self.K, diff_in)  # (g, nbOut)
        muTmp = self.MuOut + muTmp_t.T  # (nbOut, g)
        
        # Weighted sum for expected_data
        expected_data = np.einsum('g,og->o', priorsOutput, muTmp)  # (nbOut,)
        
        # ===== Vectorized conditional covariance (expSigma) =====
        # Note: np.matmul(muTmp[:,j], muTmp[:,j].T) for 1D arrays computes inner product (scalar)
        # So we compute inner product for each Gaussian
        muTmp_inner = np.einsum('og,og->g', muTmp, muTmp)  # (g,) - inner product per Gaussian
        
        # SigmaTmp_all: (g, nbOut, nbOut), transpose to (nbOut, nbOut, g)
        SigmaTmp_t = np.transpose(self.SigmaTmp_all, (1, 2, 0))  # (nbOut, nbOut, g)
        # Add scalar inner product (broadcasts to all elements)
        cov_term = SigmaTmp_t + muTmp_inner  # (nbOut, nbOut, g)
        
        # Weighted sum over Gaussians
        expSigma = np.einsum('g,opg->op', priorsOutput, cov_term)
        
        # Subtract expected inner product (matching slow version's np.matmul behavior for 1D)
        expected_inner = np.dot(expected_data, expected_data)  # scalar
        expSigma = expSigma - expected_inner + np.eye(self.nbOut) * self.diagRegFact
        
        return expected_data,priorsOutput,expSigma
    def condition_repro_gmr(self,DataIn,sample:Sample,last_input_index,last_output_index,DS,new_dt):
        if self.version=="fast" or self.version=="compare":
            if len(DataIn.shape) == 1:
                DataIn = np.expand_dims(DataIn, axis=-1)
            nbData = DataIn.shape[0]

            # prep the output arrays 
            self.conditionTPGMM(sample.params,last_input_index,last_output_index)
            expected_data = np.zeros((self.nbOut, nbData))
            expSigma = np.zeros((self.nbOut, self.nbOut, nbData))
            priorsOutput = np.zeros((self.num_of_gauss, nbData))
            
            for t in range(nbData):
                expected_data_sample,priorsOutput_sample,expSigma_sample = self.GMR_sample(DataIn[t, :])
                priorsOutput[:, t] = priorsOutput_sample
                expected_data[:, t] = expected_data_sample
                expSigma[:, :, t] = expSigma_sample
        if DS: 
            _,expected_dyn_data = self.dynSysControl(DataIn,expected_data,sample.params,last_input_index,last_output_index,new_dt)
        else:
            expected_dyn_data = None
        return expected_data.T,expSigma,priorsOutput,expected_dyn_data