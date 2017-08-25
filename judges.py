class JudgePanels(HEODAgentSociety):
    N, K, P = 18, 5, 14  # number of agents, internal dimension and number of issues
    panel_compositions = ('Aaa', 'Aab', 'Abb', 'Bbb', 'Bba', 'Baa')  # types of panel by political affiliation
    panel_indices = vstack(split(arange(N), len(panel_compositions)))  # agent indices for each panel
    panel_map = {c: i for (c, i) in zip(panel_compositions, panel_indices)}
    focal_agents = list(range(0,18,3))  # the agent indices corresponding to the capital letters
    law_vector = ones(K)/norm(ones(K))  # vector repesenting the kwoledge of the law
    party_vector = r_[-1.,-1.,0.,1.,1.]  # vector representing the political attitude
    party_vector /= norm(party_vector) 
    interaction_counter = 0
    
    def __init__(self, alpha_law, alpha_party, alpha_person, mu0XX, mu0XY, c0A, c0B, s20, *args, **params):
        """Society of agents representing panels of judges apointed by presidents 
        of political affiliation A or B.
          The society has 18 agents interacting in groups of 3, representing the 6 
        possible composition panels: Aaa, Aab, Abb, Bbb, Bba, Baa.
          The notation Xyz is to be understood as the focal agent has political
        affiliation X in a panel with agents with political affiliations y and z.
          The initial opinion state for the judges is composed by a law component,
        representing the knowledge of the law and common to all agents, a party
        component, represent the political attitude and opposite for A and B, and
        a personality component of random charater.
          To initialize the society, the weights `alpha_law`, `alpha_party` and 
        `alpha_person` are used to build the opinion vectors.
          The opinion uncertainty is a party dependent multiple of the identity 
        matrix, with the multiples being the parameters `c0A` and `c0B`.
          For the distrust, we use two paramenters `mu0XX` and `mu0XY` to characterize
        the distrust attributed between members of party X to members of X and Y,
        respectively.
          Last, the distrust uncertainty is equal to everyone, being a multiple of the
        matrix will all entries equal to 1 and given by the parameter `s20`.
        
        Inputs:
        =======
          - alpha_law, alpha_party, alpha_person: floats - weights to initial opinion
          vector components of law, party and personality.
          - mu0XX, mu0XY: floats - intra and extra-party initial distrust
          - c0A, c0B: positive floats - initial opinion uncertainties for each party
          - s20: positive float - initial distrust uncertainty
          
        API:
        ====
          Although you can use any method inside this object, it intended to be used
          a iterator over the dynamics, so we only give a description for the 
          methods provided to this end:
          - update(*args, **param): Method to make a random move using the Entropic
          Dynamics of Learning. The Options section covers the parameters recognized
          - reset(): Method to reset the system to its initial state.
          - observables: Propery to compute and return all the observables in this
          system. Notice that observable denotes a quantity of interest, not anything
          readable in memory.
          
          Options:
          --------
          opinion_norm: positive float; default is 1.0 - gives the norm for the agents' 
          opinion vectors.
          distrust_bound: positive float; defautl is 1.0 - gives the bound for the
          interval containing the distrust proxies of the agents.
          constants: tuple with acceptable values 'norm', 'C', 's2', 'bounds'; default 
          is ('C', 'norm') - each value
          keeps the respective variable fixed under the dynamics, where 'norm' is for the 
          opinion vectors, 'C' is for the opinion uncertainty, 's2' is for the distrust
          uncertainty and 'bounds' is for the distrust interval.
        """
        assert (c0A >= 0) and (c0B >= 0) and (s20 >= 0)
        self.alpha_law = alpha_law
        self.alpha_party = alpha_party
        self.alpha_person = alpha_person
        self.mu0XX, self.mu0XY = mu0XX, mu0XY
        self.c0A, c0B = c0A, c0B
        self.s20 = s20
        
        # Building the initial opinion vectors
        #             law component                 party compoenent
        A = alpha_law*self.law_vector + alpha_party*self.party_vector
        B = alpha_law*self.law_vector - alpha_party*self.party_vector
        #            A a a A a b A b b B b b B b a B a a                  personality component
        w0 = vstack([A,A,A,A,A,B,A,B,B,B,B,B,B,B,A,B,A,A]) + alpha_person*randn(self.N,self.K)
        w0 /= row_norm(w0)
        
        # Building the initial opinion uncertainties
        I = eye(self.K)
        CA, CB = c0A*I, c0B*I
        #            A  a  a  A  a  b  A  b  b  B  b  b  B  b  a  B  a  a
        C0 = stack([CA,CA,CA,CA,CA,CB,CA,CB,CB,CB,CB,CB,CB,CB,CA,CB,CA,CA], axis=0)
        
        # Building the initial distrust proxies - X in (A,B), Y in (A,B), i, j, k are agents
        #                      i     j     k
        muXxx_tile = array([[-10, mu0XX, mu0XX],   # ii==XX, ij==XX, ik==XX
                            [mu0XX, -10, mu0XX],   # ji==XX, jj==XX, jk==XX
                            [mu0XX, mu0XX, -10]])  # ki==XX, kj==XX, kk==XX
        muXxy_tile = array([[-10, mu0XX, mu0XY],   # ii==XX, ij==XX, ik==XY
                            [mu0XX, -10, mu0XY],   # ji==XX, jj==XX, jk==XY
                            [mu0XY, mu0XY, -10]])  # ki==XY, kj==XY, kk==XX
        muXyy_tile = array([[-10, mu0XY, mu0XY],   # ii==XX, ij==XY, ik==XY
                            [mu0XY, -10, mu0XX],   # ji==XY, jj==XX, jk==XX
                            [mu0XY, mu0XX, -10]])  # ki==XY, kj==XX, kk==XX
        mu_map = {c: mt for (c, mt) in zip(self.panel_compositions,[muXxx_tile, muXxy_tile, muXyy_tile]*2)}
        mu0 = zeros((self.N, self.N))
        panel_pairs = stack([vstack([[i,j] for i in p for j in p]) for p in self.panel_indices], axis=0)
        for panel, indices in self.panel_map.items():
            pair_indices = vstack([[i,j] for i in indices for j in indices])
            I, J = split(pair_indices, 2, axis=1)
            mu0[I, J] = mu_map[panel].reshape(I.shape)
            
        # Building the initial distrust uncertainty
        s20 = s20*ones((self.N, self.N))
        fill_diagonal(s20, 0.0)
        
        # let the super class prepare the remaining of the object
        super().__init__(w0, C0, mu0, s20)
        
    @property
    def issue_list(self):
        if not hasattr(self, '_issue_list'):
            thetas = linspace(-pi/2,pi/2, self.P)
            self._issue_list = vstack([cos(t)*self.law_vector + sin(t)*self.party_vector
                                        for t in thetas])
        return self._issue_list
    
    def pick_issue(self, *args, **params):
        k = choice(len(self.issue_list), size=6, replace=True)
        x = self.issue_list[k]
        return x

    def pick_agents(self, *args, **params):
        pairs = vstack([choice(p, size=2, replace=False)
                        for p in self.panel_indices])
        return pairs

    def update(self, *args, constants=('norm', 'C'), **params): 
        xs = self.pick_issue(*args, **params)
        pairs = self.pick_agents(*args, **params)
        for ((i,j),x) in zip(pairs, xs):
            self.interaction(i, j, x, *args, **params)
            
    @property
    def votes(self):
        if not hasattr(self, '_votes'):     
            h = self.w[self.focal_agents, :]@self.issue_list.T
            self._votes = (self.panel_compositions, h/row_norm(h))
        return self._votes
    
    @property
    def panel_overlaps(self):
        if not hasattr(self, '_panel_overlaps'):
            panels, votes = self.votes
            overlaps = votes@votes.T
            panel_pairs = array([[' '.join([p1,p2]) for p1 in panels] for p2 in panels])
            self._panel_overlaps = (panel_pairs, overlaps)
        return self._panel_overlaps
        
    @property
    def observables(self):
        if not hasattr(self, '_observables'):
            self._observables = 'votes panel_overlaps'.split()
        return {n:getattr(self, n) for n in self._observables}