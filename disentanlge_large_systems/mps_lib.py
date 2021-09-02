import numpy as np 




def chip_site(j,lmbda_prev,theta,chi_vec,d=2):

	X,Lambda,Y = np.linalg.svd( theta.reshape(chi_vec[j]*d, -1), full_matrices=False)

	Lambda = Lambda[:chi_vec[j+1]]/np.sqrt(np.sum(np.abs(Lambda[:chi_vec[j+1]])**2))
	X = X[:,:chi_vec[j+1]].reshape(chi_vec[j]  ,d,chi_vec[j+1])	
	Y = Y[:chi_vec[j+1],:]

	Gamma = np.einsum('a,aib->aib',  np.divide(1.0, lmbda_prev, out=np.zeros_like(lmbda_prev), where=np.abs(lmbda_prev)>=1E-14), X.reshape(chi_vec[j], d, chi_vec[j+1]), )
	theta = np.einsum('a,as->as',Lambda,Y)

	return Gamma, Lambda, theta


def ED_to_MPS(psi,chi_vec,L,d=2):

	# initialize MPS data
	Gammas =[]
	Lambdas=[np.array([1.0,]),]
	theta=psi.copy()

	for j in range(L):
		G, L, theta = chip_site(j,Lambdas[j],theta,chi_vec,d=d)
		Gammas.append(G)
		Lambdas.append(L)

	return Gammas, Lambdas


def MPS_to_ED(G, Lambdas, canonical=0):
	"""
	## example for L=4, 

	psi_MPS = np.einsum('a,aib,b,bjc,c,ckd,d,dle,e->ijkl', Lambdas[0],Gammas[0],Lambdas[1],Gammas[1],Lambdas[2],Gammas[2],Lambdas[3],Gammas[3],Lambdas[4] ).reshape(-1,)

	"""

	if canonical==0: # canonical form
		psi_MPS=Lambdas[0]
		for j in range(len(G)):
			psi_MPS = np.einsum('...a,aib,b->...ib', psi_MPS, G[j],Lambdas[j+1], )
	
	else: # left or right canonical form
		psi_MPS=np.array([1.0])
		for j in range(len(G)):
			psi_MPS = np.einsum('...a,aib->...ib', psi_MPS, G[j] )

	return psi_MPS.reshape(-1,)


def left_canonical(Gammas,Lambdas):
	As=[]
	for j in range(len(Gammas)):
		A=np.einsum('a,aib->aib',Lambdas[j],Gammas[j])
		As.append(A)
	return As

def right_canonical(Gammas,Lambdas):
	Bs=[]
	for j in range(len(Gammas)):
		B=np.einsum('aib,b->aib',Gammas[j],Lambdas[j+1])
		Bs.append(B)
	return Bs

def canonical(G,Lambdas,L,chi_vec,canonical=-1):
	psi_MPS=MPS_to_ED(G, Lambdas, canonical=canonical)
	return ED_to_MPS(psi_MPS,chi_vec,L)



def MPS_norm(G,Lambdas,canonical=0):
	norm=1.0

	if canonical==0: # canonical form
		for j in range(len(G)-1):
			theta = np.einsum('a,aib,b->aib',Lambdas[j],G[j],Lambdas[j+1],)
			norm*=np.sum(np.abs(theta)**2)
		#norm/=len(G)-1
	elif canonical==+1: # right canonical form
		# psi_MPS=np.array([1.0])
		# for j in range(len(G)):
		# 	psi_MPS = np.einsum('...a,aib->...ib', psi_MPS, G[j] )
		# 	norm*=np.sum(np.abs(psi_MPS)**2)
		theta=np.eye(2)/2
		for j in range(len(G)):
			theta = np.einsum('ab,aic,bid->cd', theta, G[j].conj(), G[j] )
		norm=theta.squeeze()
	else: # left canonical form
		theta=np.eye(2)/2
		for j in range(len(G)-1,-1,-1):	
			theta = np.einsum('aib,cid,bd->ac', G[j].conj(), G[j], theta )
		norm=theta.squeeze()

	return np.sqrt(norm)


def random_MPS(L,d,chi_vec):

	As=[]
	Bs=[]
	Lambdas=[np.array([1.0])]

	for j in range(L):

		M=np.random.normal(size=(chi_vec[j]*d,chi_vec[j+1]) )
		X, S, Y = np.linalg.svd(M, full_matrices=False)

		X=X.reshape(chi_vec[j],d,chi_vec[j+1])

		As.append( np.einsum('a,aib->aib',Lambdas[j],X) )
		Lambdas.append( S[:chi_vec[j+1]]/np.sqrt(np.sum(np.abs(S[:chi_vec[j+1]])**2)) )
		Bs.append( np.einsum('aib,b->aib',X,Lambdas[j+1]) )

	# normalize

	norm_n=MPS_norm(As,Lambdas,canonical=-1)
	norm_p=MPS_norm(Bs,Lambdas,canonical=1)

	for j in range(L):
		As[j]/=norm_n**(1.0/L)
		Bs[j]/=norm_p**(1.0/L)


	return As, Bs, Lambdas


