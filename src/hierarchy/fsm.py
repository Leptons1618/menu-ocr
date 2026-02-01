"""
Finite-State Machine for enforcing valid label transitions.
Uses Viterbi algorithm for sequence decoding.
"""

import numpy as np
from typing import Optional
from ..models.schema import TextElementType


class HierarchyFSM:
    """
    Finite-state machine enforcing valid menu hierarchy transitions.
    
    Valid transitions ensure logical menu structure:
    - Section headers can start new sections or contain groups/items
    - Group headers contain items
    - Items can have prices and descriptions
    - No invalid jumps (e.g., description before any item)
    """
    
    # Define valid transitions
    # Key: current state, Value: set of valid next states
    TRANSITIONS = {
        'START': {
            TextElementType.SECTION_HEADER,
            TextElementType.GROUP_HEADER,
            TextElementType.ITEM_NAME,
            TextElementType.OTHER,
        },
        TextElementType.SECTION_HEADER: {
            TextElementType.SECTION_HEADER,
            TextElementType.GROUP_HEADER,
            TextElementType.ITEM_NAME,
            TextElementType.ITEM_DESCRIPTION,  # Section description
            TextElementType.OTHER,
        },
        TextElementType.GROUP_HEADER: {
            TextElementType.SECTION_HEADER,
            TextElementType.GROUP_HEADER,
            TextElementType.ITEM_NAME,
            TextElementType.ITEM_DESCRIPTION,  # Group description
            TextElementType.OTHER,
        },
        TextElementType.ITEM_NAME: {
            TextElementType.SECTION_HEADER,
            TextElementType.GROUP_HEADER,
            TextElementType.ITEM_NAME,
            TextElementType.ITEM_PRICE,
            TextElementType.ITEM_DESCRIPTION,
            TextElementType.METADATA,
            TextElementType.OTHER,
        },
        TextElementType.ITEM_PRICE: {
            TextElementType.SECTION_HEADER,
            TextElementType.GROUP_HEADER,
            TextElementType.ITEM_NAME,
            TextElementType.ITEM_PRICE,  # Multiple prices (variants)
            TextElementType.ITEM_DESCRIPTION,
            TextElementType.METADATA,
            TextElementType.OTHER,
        },
        TextElementType.ITEM_DESCRIPTION: {
            TextElementType.SECTION_HEADER,
            TextElementType.GROUP_HEADER,
            TextElementType.ITEM_NAME,
            TextElementType.ITEM_PRICE,
            TextElementType.ITEM_DESCRIPTION,  # Multi-line description
            TextElementType.METADATA,
            TextElementType.OTHER,
        },
        TextElementType.METADATA: {
            TextElementType.SECTION_HEADER,
            TextElementType.GROUP_HEADER,
            TextElementType.ITEM_NAME,
            TextElementType.ITEM_PRICE,
            TextElementType.ITEM_DESCRIPTION,
            TextElementType.METADATA,
            TextElementType.OTHER,
        },
        TextElementType.OTHER: {
            TextElementType.SECTION_HEADER,
            TextElementType.GROUP_HEADER,
            TextElementType.ITEM_NAME,
            TextElementType.ITEM_PRICE,
            TextElementType.ITEM_DESCRIPTION,
            TextElementType.METADATA,
            TextElementType.OTHER,
        },
    }
    
    # Transition costs (lower = more natural transition)
    TRANSITION_COSTS = {
        # Natural progressions have zero cost
        (TextElementType.SECTION_HEADER, TextElementType.GROUP_HEADER): 0.0,
        (TextElementType.SECTION_HEADER, TextElementType.ITEM_NAME): 0.0,
        (TextElementType.GROUP_HEADER, TextElementType.ITEM_NAME): 0.0,
        (TextElementType.ITEM_NAME, TextElementType.ITEM_PRICE): 0.0,
        (TextElementType.ITEM_NAME, TextElementType.ITEM_DESCRIPTION): 0.0,
        (TextElementType.ITEM_NAME, TextElementType.ITEM_NAME): 0.0,
        (TextElementType.ITEM_PRICE, TextElementType.ITEM_NAME): 0.0,
        (TextElementType.ITEM_DESCRIPTION, TextElementType.ITEM_NAME): 0.0,
        
        # Slight cost for section changes
        (TextElementType.ITEM_NAME, TextElementType.SECTION_HEADER): 0.1,
        (TextElementType.ITEM_NAME, TextElementType.GROUP_HEADER): 0.1,
        (TextElementType.ITEM_PRICE, TextElementType.SECTION_HEADER): 0.1,
        (TextElementType.ITEM_PRICE, TextElementType.GROUP_HEADER): 0.1,
        
        # Default cost for other transitions
    }
    DEFAULT_TRANSITION_COST = 0.2
    INVALID_TRANSITION_COST = 10.0
    
    def __init__(self):
        """Initialize FSM."""
        self.states = list(TextElementType)
    
    def is_valid_transition(
        self,
        from_state: TextElementType,
        to_state: TextElementType,
    ) -> bool:
        """Check if transition is valid."""
        valid_next = self.TRANSITIONS.get(from_state, set())
        return to_state in valid_next
    
    def get_transition_cost(
        self,
        from_state: TextElementType,
        to_state: TextElementType,
    ) -> float:
        """Get cost of transition."""
        if not self.is_valid_transition(from_state, to_state):
            return self.INVALID_TRANSITION_COST
        
        return self.TRANSITION_COSTS.get(
            (from_state, to_state),
            self.DEFAULT_TRANSITION_COST
        )
    
    def viterbi_decode(
        self,
        observations: list[dict[TextElementType, float]],
        use_log_probs: bool = True,
    ) -> list[TextElementType]:
        """
        Find most likely label sequence respecting transitions.
        
        Uses Viterbi algorithm to find optimal path through state space.
        
        Parameters:
        -----------
        observations : List of dicts mapping states to scores/probabilities
                      Higher score = more likely for that state
        use_log_probs : If True, treat scores as log probabilities
        
        Returns:
        --------
        List of TextElementType labels, one per observation
        """
        if not observations:
            return []
        
        n = len(observations)
        states = self.states
        n_states = len(states)
        
        # Convert to numpy for efficiency
        # V[t, s] = best score to reach state s at time t
        V = np.full((n, n_states), -np.inf if use_log_probs else 0.0)
        # Backpointer for path reconstruction
        backptr = np.zeros((n, n_states), dtype=int)
        
        # State to index mapping
        state_idx = {s: i for i, s in enumerate(states)}
        
        # Initialize first position
        valid_starts = self.TRANSITIONS['START']
        for s in valid_starts:
            idx = state_idx[s]
            score = observations[0].get(s, -10.0 if use_log_probs else 0.01)
            V[0, idx] = score
        
        # Forward pass
        for t in range(1, n):
            for s_idx, s in enumerate(states):
                obs_score = observations[t].get(s, -10.0 if use_log_probs else 0.01)
                
                best_score = -np.inf if use_log_probs else 0.0
                best_prev = 0
                
                for prev_idx, prev_s in enumerate(states):
                    if V[t-1, prev_idx] == (-np.inf if use_log_probs else 0.0):
                        continue
                    
                    # Transition score (negative cost for log probs)
                    trans_cost = self.get_transition_cost(prev_s, s)
                    if use_log_probs:
                        trans_score = -trans_cost
                    else:
                        trans_score = np.exp(-trans_cost)
                    
                    # Total score
                    if use_log_probs:
                        score = V[t-1, prev_idx] + trans_score + obs_score
                    else:
                        score = V[t-1, prev_idx] * trans_score * obs_score
                    
                    if score > best_score:
                        best_score = score
                        best_prev = prev_idx
                
                V[t, s_idx] = best_score
                backptr[t, s_idx] = best_prev
        
        # Backtrack to find best path
        path = []
        
        # Find best final state
        best_final_idx = np.argmax(V[n-1])
        path.append(states[best_final_idx])
        
        # Backtrack
        current_idx = best_final_idx
        for t in range(n-1, 0, -1):
            current_idx = backptr[t, current_idx]
            path.append(states[current_idx])
        
        path.reverse()
        return path
    
    def apply_constraints(
        self,
        predictions: list[tuple[TextElementType, float]],
    ) -> list[TextElementType]:
        """
        Apply FSM constraints to a list of predictions.
        
        Converts (label, confidence) pairs to observation dicts
        and runs Viterbi decoding.
        
        Parameters:
        -----------
        predictions : List of (label, confidence) tuples from classifier
        
        Returns:
        --------
        List of labels respecting transition constraints
        """
        # Convert predictions to observation dicts
        observations = []
        for label, conf in predictions:
            obs = {s: -5.0 for s in self.states}  # Low base score
            obs[label] = np.log(max(conf, 0.01))  # High score for predicted label
            
            # Add some probability mass to similar labels
            if label == TextElementType.SECTION_HEADER:
                obs[TextElementType.GROUP_HEADER] = np.log(0.2)
            elif label == TextElementType.GROUP_HEADER:
                obs[TextElementType.SECTION_HEADER] = np.log(0.15)
            elif label == TextElementType.ITEM_NAME:
                obs[TextElementType.ITEM_DESCRIPTION] = np.log(0.1)
            
            observations.append(obs)
        
        return self.viterbi_decode(observations)
