import heapq
from .data import InstMap

import torch


class SddNode:
    """Sentential Decision Diagram (SDD)"""

    # typedef's
    FALSE, TRUE, LITERAL, DECOMPOSITION = 0, 1, 2, 3

    ########################################
    # CONSTRUCTOR + BASIC FUNCTIONS
    ########################################

    def __init__(self, node_type, alpha, vtree, manager):
        """Constructor

        node_type is FALSE, TRUE, LITERAL or DECOMPOSITION
        alpha is a literal if node_type is LITERAL
        alpha is a list of elements if node_type is DECOMPOSITION
        vtree is the vtree that the node is normalized for
        manager is the SDD manager"""

        self.node_type = node_type
        self.vtree = vtree
        if self.is_false() or self.is_true():
            self.literal, self.elements = None, None
        elif self.is_literal():
            self.literal, self.elements = alpha, None
        else:  # self.is_decomposition()
            self.literal, self.elements = None, alpha
        if manager is None:
            self.id = None
        else:
            self.id = manager.new_id()
        self.data = None  # data field
        self._bit = False  # internal bit field
        self._array = None  # internal array field
        # this is needed by normalized SDDs later
        self.is_false_sdd = node_type == SddNode.FALSE

    def __repr__(self, use_index=False):
        if use_index:
            index = lambda n: n.index
        else:
            index = lambda n: n.id
        if self.is_false():
            st = 'F %d' % index(self)
        elif self.is_true():
            st = 'T %d' % index(self)
        elif self.is_literal():
            st = 'L %d %d %d' % (index(self), self.vtree.id, self.literal)
        else:  # self.is_decomposition()
            els = self.elements
            st_el = " ".join('%d %d' % (index(p), index(s)) for p, s in els)
            st = 'D %d %d %d %s' % (index(self), self.vtree.id, len(els), st_el)
        return st

    @classmethod
    def _dummy_true(cls):
        """used for enumeration"""
        return cls(SddNode.TRUE, None, None, None)

    def is_false(self):
        """Returns true if node is FALSE, and false otherwise"""
        return self.node_type is SddNode.FALSE

    def is_true(self):
        """Returns true if node is TRUE, and false otherwise"""
        return self.node_type is SddNode.TRUE

    def is_literal(self):
        """Returns true if node is LITERAL, and false otherwise"""
        return self.node_type is SddNode.LITERAL

    def is_decomposition(self):
        """Returns true if node is DECOMPOSITION, and false otherwise"""
        return self.node_type is SddNode.DECOMPOSITION

    def count(self):
        """Returns the number of decision nodes in the SDD"""
        return sum(1 for n in self if n.is_decomposition())

    def size(self):
        """Returns the aggregate size of decision nodes in the SDD"""
        return sum(len(n.elements) for n in self if n.is_decomposition())

    def _node_count(self):
        """Returns the number of decision and terminal nodes in the SDD"""
        return sum(1 for n in self)

    ########################################
    # TRAVERSAL
    ########################################

    def __iter__(self, first_call=True, clear_data=False):
        """post-order (children before parents) generator"""
        if self._bit: return
        self._bit = True

        if self.is_decomposition():
            for p, s in self.elements:
                for node in p.__iter__(first_call=False): yield node
                for node in s.__iter__(first_call=False): yield node
        yield self

        if first_call:
            self.clear_bits(clear_data=clear_data)

    def post_order(self, clear_data=False):
        """post-order generator"""
        return self.__iter__(first_call=True, clear_data=clear_data)

    def pre_order(self, first_call=True, clear_data=False):
        """pre-order generator"""
        if self._bit: return
        self._bit = True

        yield self
        if self.is_decomposition():
            for p, s in self.elements:
                for node in p.pre_order(first_call=False): yield node
                for node in s.pre_order(first_call=False): yield node

        if first_call:
            self.clear_bits(clear_data=clear_data)

    def clear_bits(self, clear_data=False):
        """Recursively clears bits.  For use when recursively navigating an
        SDD by marking bits (not for use with SddNode.as_list).

        Set clear_data to True to also clear data."""
        if self._bit is False: return
        self._bit = False
        if clear_data: self.data = None
        if self.is_decomposition():
            for p, s in self.elements:
                p.clear_bits(clear_data=clear_data)
                s.clear_bits(clear_data=clear_data)

    def as_list(self, reverse=False, clear_data=True):
        """iterating over an SDD's nodes, as a list

        This is faster than recursive traversal of an SDD, as done in
        __iter__() and pre_order().  If clear_data=True, then the data
        fields of each node is reset to None.  If reverse=True, then
        the list is traversed backwards (parents before children).

        Examples
        --------

        >>> for i,node in enumerate(self.as_list()):
        ...     node.data = i

        """
        self._linearize()
        if reverse:
            for node in reversed(self._array):
                yield node
        else:
            for node in self._array:
                yield node
        if clear_data:
            for node in self._array:
                node.data = None

    def _linearize(self):
        """linearize SDD"""
        if self._array is not None: return  # already linearized
        self._array = list(self)

    def _is_bits_and_data_clear(self):
        """sanity check, for testing"""
        for node in self.as_list(clear_data=False):
            if node._bit is not False or \
                    node.data is not None:
                return False
        return True

    ########################################
    # QUERIES
    ########################################

    def model_count(self, vtree):
        """Compute model count of SDD relative to vtree.

        It is much faster to linearize the SDD compared to computing
        the model count recursively, if model counting may times."""
        if self.is_false():   return 0
        if self.is_true():    return 1 << vtree.var_count
        if self.is_literal(): return 1 << (vtree.var_count - 1)
        if not self.vtree.is_sub_vtree_of(vtree):
            msg = "node vtree is not sub-vtree of given vtree"
            raise AssertionError(msg)

        # self is decomposition node
        for node in self.as_list(clear_data=True):
            if node.is_false():
                count = 0
            elif node.is_true():
                count = 1
            elif node.is_literal():
                count = 1
            else:  # node.is_decomposition()
                count = 0
                left_vc = node.vtree.left.var_count
                right_vc = node.vtree.right.var_count
                for prime, sub in node.elements:
                    if sub.is_false(): continue
                    prime_vc = 0 if prime.is_true() else prime.vtree.var_count
                    sub_vc = 0 if sub.is_true() else sub.vtree.var_count
                    left_gap = left_vc - prime_vc
                    right_gap = right_vc - sub_vc
                    prime_mc = prime.data << left_gap
                    sub_mc = sub.data << right_gap
                    count += prime_mc * sub_mc
            node.data = count
        gap = (vtree.var_count - node.vtree.var_count)
        return count << gap

    def is_model_marker(self, inst, clear_bits=True, clear_data=True):
        """Returns None if inst is not a model, otherwise it returns the
        value/element that is satisfied.

        inst should be of type Inst or InstMap.

        Performs recursive test, which can be faster than linear
        traversal as in SddNode.as_list.

        If clear_data is set to false, then each node.data will point
        to the value/element that satisfies the SDD (if one exists).

        clear_bits is for internal use."""
        if self._bit: return self.data
        self._bit = True

        if self.is_false():
            is_model = None
        elif self.is_true():
            if hasattr(self.vtree, 'var'):  # normalized SDD
                is_model = inst[self.vtree.var]
            else:  # trimmed SDD
                is_model = True  # AC: not normalized, self.data = ?
        elif self.is_literal():
            if inst.is_compatible(self.literal):
                is_model = inst[self.vtree.var]
            else:
                is_model = None
        else:  # self.is_decomposition()
            is_model = None
            for p, s in self.elements:
                if s.is_false_sdd: continue  # for normalized SDDs
                if p.is_model_marker(inst, clear_bits=False) is None: continue
                if s.is_model_marker(inst, clear_bits=False) is None: continue
                is_model = (p, s)
                break
        self.data = is_model

        if clear_bits:
            self.clear_bits(clear_data=clear_data)
        return is_model

    def is_model(self, inst):
        """Returns True if inst (of type Inst or InstMap) is a model
        of the SDD, and False otherwise."""
        return self.is_model_marker(inst) is not None

    def models(self, vtree, lexical=False):
        """A generator for the models of an SDD.
        
        If lexical is True, then models will be given in lexical
        (sorted) order.  This is typically slower than when
        lexical=False."""

        iterator = self._models_lexical(vtree) if lexical \
            else self._models_recursive(vtree)
        return iterator

    def _models_recursive(self, vtree):
        """Recursive model enumeration"""

        if self.is_false():
            return
        elif vtree.is_leaf():
            if self.is_true():
                yield InstMap.from_literal(-vtree.var)
                yield InstMap.from_literal(vtree.var)
            elif self.is_literal():
                yield InstMap.from_literal(self.literal)
        elif self.vtree == vtree:
            for prime, sub in self.elements:
                if sub.is_false_sdd: continue  # for normalized SDDs
                for left_model in prime._models_recursive(vtree.left):
                    for right_model in sub._models_recursive(vtree.right):
                        yield left_model.concat(right_model)
        else:  # there is a gap
            true_node = SddNode._dummy_true()
            if self.is_true():
                prime, sub = true_node, true_node
            # elif self.vtree.is_sub_vtree_of(vtree.left):
            elif self.vtree.id < vtree.id:
                prime, sub = self, true_node
            # elif self.vtree.is_sub_vtree_of(vtree.right):
            elif self.vtree.id > vtree.id:
                prime, sub = true_node, self
            else:
                msg = "node vtree is not sub-vtree of given vtree"
                raise AssertionError(msg)
            for left_model in prime._models_recursive(vtree.left):
                for right_model in sub._models_recursive(vtree.right):
                    yield left_model.concat(right_model)

    def _models_lexical(self, vtree):
        """Lexical model enumeration"""
        enum = SddEnumerator(vtree)
        return enum.enumerator(self)

    def minimum_cardinality(self):
        "AC: NEED TO TEST"
        """Computes the minimum cardinality model of an SDD."""
        if self.is_false():   return 0
        if self.is_true():    return 0
        if self.is_literal(): return 1 if self.literal > 0 else 0

        # self is decomposition node
        for node in self.as_list(clear_data=True):
            if node.is_false():
                card = 0
            elif node.is_true():
                card = 0
            elif node.is_literal():
                card = 1 if node.literal > 0 else 0
            else:  # node.is_decomposition()
                card = None
                for prime, sub in node.elements:
                    if sub.is_false(): continue
                    element_card = prime.data + sub.data
                    if card is None:
                        card = element_card
                    else:
                        card = min(card, element_card)
            node.data = card
        return card

    def __lt__(self, other):
        return self.id < other.id


########################################
# NORMALIZED SDDS
########################################

class NormalizedSddNode(SddNode):
    """Normalized Sentential Decision Diagram (SDD)"""

    def __init__(self, node_type, alpha, vtree, manager):
        """Constructor

        node_type is FALSE, TRUE, LITERAL or DECOMPOSITION
        alpha is a literal if node_type is LITERAL
        alpha is a list of elements if node_type is DECOMPOSITION
        vtree is the vtree that the node is normalized for
        manager is the PSDD manager"""

        SddNode.__init__(self, node_type, alpha, vtree, manager)
        self.negation = None
        # self.is_false_sdd = node_type == SddNode.FALSE
        if node_type is SddNode.DECOMPOSITION:
            self.positive_elements = \
                tuple((p, s) for p, s in self.elements if not s.is_false_sdd)
        self._positive_array = None

    def _positive_node_count(self):
        """Returns the number of (decision and terminal) nodes in the SDD"""
        count = 0
        for node in self.positive_iter():
            count += 1
        return count

    def positive_iter(self, first_call=True, clear_data=False):
        """post-order (children before parents) generator, skipping false SDD
        nodes"""
        if self._bit: return
        if self.is_false_sdd: return
        self._bit = True

        if self.is_decomposition():
            for p, s in self.elements:
                if s.is_false_sdd: continue
                for node in p.positive_iter(first_call=False): yield node
                for node in s.positive_iter(first_call=False): yield node
        yield self

        if first_call:
            self.clear_bits(clear_data=clear_data)

    def as_positive_list(self, reverse=False, clear_data=True):
        """iterating over an SDD's nodes, as a list.  See SddNode.as_list"""
        if self._positive_array is None:
            self._positive_array = list(self.positive_iter())
        if reverse:
            for node in reversed(self._positive_array):
                yield node
        else:
            for node in self._positive_array:
                yield node
        if clear_data:
            for node in self._positive_array:
                node.data = None

    def model_count(self, evidence=InstMap(), clear_data=True):
        """Compute model count of a normalized SDD.

        SddNode.model_count does not assume the SDD is normalized."""
        for node in self.as_list(clear_data=clear_data):
            if node.is_false():
                count = 0
            elif node.is_true():
                count = 1 if node.vtree.var in evidence else 2
            elif node.is_literal():
                count = 1 if evidence.is_compatible(node.literal) else 0
            else:  # node.is_decomposition()
                count = sum(p.data * s.data for p, s in node.elements)
            node.data = count

        return count

    def get_weighted_mpe(self, lit_weights, clear_data=True):
        """Compute the MPE instation given weights associated with literals.

        Assumes the SDD is normalized.
        """
        for node in self.as_positive_list(clear_data=clear_data):
            if node.is_false():
                # No configuration on false
                data = (0, [])
            elif node.is_true():
                # Need to pick max assignment for variable here
                b_ind = max([0, 1], key=lambda x: lit_weights[node.vtree.var - 1][x])
                # If it's a 0, then -lit number, else lit number
                data = (lit_weights[node.vtree.var - 1][b_ind], [pow(-1, b_ind + 1) * node.vtree.var])
            elif node.is_literal():
                if node.literal > 0:
                    data = (lit_weights[node.literal - 1][1], [node.literal])
                else:
                    data = (lit_weights[-node.literal - 1][0], [node.literal])
            else:  # node is_decomposition()
                data = max(((p.data[0] * s.data[0], p.data[1] + s.data[1]) for p, s in node.positive_elements)
                           , key=lambda x: x[0])
            node.data = data
        return data

    def weighted_model_count(self, lit_weights, clear_data=True):
        """ Compute weighted model count given literal weights

        Assumes the SDD is normalized.
        """
        for node in self.as_list(clear_data=clear_data):
            if node.is_false():
                data = 0
            elif node.is_true():
                data = 1
            elif node.is_literal():
                if node.literal > 0:
                    data = lit_weights[node.literal - 1][1]
                else:
                    data = lit_weights[-node.literal - 1][0]
            else:  # node is_decomposition
                data = sum(p.data * s.data for p, s in node.elements)
            node.data = data
        return data

    def generate_pt_ac(self, trueprobs, clear_data=True):
        """
        Generates a pytorch arithmetic circuit according to the weighted model counting procedure for this SDD.
        Assumes the SDD is normalized.
        """

        true_tensor = torch.ones(1, device=trueprobs.device)
        false_tensor = torch.zeros(1, device=trueprobs.device)

        node_to_data_dict = {}

        for index, node in enumerate(self.as_list(clear_data=clear_data)):
            if node.is_false():
                data = false_tensor
            elif node.is_true():
                data = true_tensor
            elif node.is_literal():
                if node.literal > 0:
                    data = trueprobs[:, node.literal - 1]
                else:
                    data = 1.0 - trueprobs[:, -node.literal - 1]
            else:  # node.is_decomposition
                res = []
                for p, s in node.elements:
                    p_data = node_to_data_dict[p]
                    s_data = node_to_data_dict[s]
                    if p_data is false_tensor or s_data is false_tensor:
                        pass
                    elif p_data is true_tensor:
                        res.append(s_data)
                    elif s_data is true_tensor:
                        res.append(p_data)
                    else:
                        res.append(s_data * p_data)

                if len(res) == 0:
                    res.append(false_tensor)

                data = torch.stack(res, dim=0).sum(dim=0)
            node_to_data_dict[node] = data
        return data

    def generate_pt_ac_log(self, trueprobs, clear_data=True):
        """
        Generates a pytorch arithmetic circuit according to the weighted model counting procedure for this SDD.
        Assumes the SDD is normalized.
        """

        assert not torch.isnan(trueprobs).any() and not torch.isinf(trueprobs).any()
        true_tensor = torch.zeros(1, device=trueprobs.device)
        false_tensor = torch.full((1,), -float("inf"), device=trueprobs.device)
        EPS = 1e-10

        node_to_data_dict = {}

        for index, node in enumerate(self.as_list(clear_data=clear_data)):
            if node.is_false():
                data = false_tensor
            elif node.is_true():
                data = true_tensor
            elif node.is_literal():
                if node.literal > 0:
                    data = (trueprobs[:, node.literal - 1] + EPS).log()
                else:
                    data = (1.0 - trueprobs[:, -node.literal - 1] + EPS).log()
            else:  # node.is_decomposition
                res = []
                for p, s in node.elements:
                    p_data = node_to_data_dict[p]
                    s_data = node_to_data_dict[s]
                    if p_data is false_tensor or s_data is false_tensor:
                        pass
                    elif p_data is true_tensor:
                        res.append(s_data)
                    elif s_data is true_tensor:
                        res.append(p_data)
                    else:
                        res.append(s_data + p_data)

                if len(res) == 0:
                    res.append(false_tensor)

                s = torch.stack(res, dim=0)
                mask = torch.isinf(s).all(dim=0)
                assert not torch.isnan(s).any()
                data = torch.where(mask, s[0], torch.logsumexp(s, dim=0))
                # data = torch.logsumexp(s, dim=0)
            node_to_data_dict[node] = data
        return data

    def count_variables(self):
        """
        :return: Returns the total number of variables.
        """
        varset = set()
        for node in self.as_list(clear_data=True):
            if node.is_literal():
                varset.add(abs(node.literal))
        return len(varset)

    def highest_index(self):
        """
        :return: Returns the highest index (variable) in the sdd. Given x1, x2 and x3, this would result in 3, this
            indicates that in generate_tf_ac the input leaves should have at least length 3.
        """
        return max([abs(node.literal) for node in self.as_list(clear_data=True) if node.is_literal()])


########################################
# MODEL ENUMERATION
########################################

class SddEnumerator:
    """Manager for lexical model enumeration.

    Caching only nodes (caching elements may not help much?)"""

    @staticmethod
    def _element_update(element_enum, inst):
        """This is invoked after inst.concat(other)"""
        pass

    def __init__(self, vtree):
        self.vtree = vtree
        self.true = SddNode._dummy_true()
        self.node_cache = dict((v, dict()) for v in vtree)
        self.terminal_enumerator = SddTerminalEnumerator

    def lookup_node_enum(self, node, vtree):
        cache = self.node_cache[vtree]
        if node in cache:
            return cache[node].cached_iter()
        else:
            enum = SddNodeEnumerator(node, vtree, self)
            cache[node] = enum
            return enum.cached_iter()

    def enumerator(self, node):
        return SddNodeEnumerator(node, self.vtree, self)


class SddTerminalEnumerator:
    """Enumerator for terminal SDD nodes"""

    def __init__(self, node, vtree):
        self.heap = []

        if node.is_false():
            pass
        elif node.is_literal():
            inst = InstMap.from_literal(node.literal)
            heapq.heappush(self.heap, inst)
        if node.is_true():
            inst = InstMap.from_literal(-vtree.var)
            heapq.heappush(self.heap, inst)
            inst = InstMap.from_literal(vtree.var)
            heapq.heappush(self.heap, inst)

    def __iter__(self):
        return self

    def empty(self):
        return len(self.heap) == 0

    def __next__(self):
        if self.empty(): raise StopIteration()
        return heapq.heappop(self.heap)

    def __cmp__(self, other):
        return cmp(self.heap[0], other.heap[0])


class SddNodeEnumerator:
    """Enumerator for SDD decomposition nodes"""

    def __init__(self, node, vtree, enum_manager):
        self.heap = []
        self.topk = []

        if node.is_false():
            pass
        elif vtree.is_leaf():
            enum = enum_manager.terminal_enumerator(node, vtree)
            if not enum.empty(): heapq.heappush(self.heap, enum)
        elif node.vtree == vtree:  # node.is_decomposition
            for prime, sub in node.elements:  # initialize
                # if sub.is_false(): continue
                if sub.is_false_sdd: continue
                enum = SddElementEnumerator(prime, sub, node, vtree, enum_manager)
                if not enum.empty(): heapq.heappush(self.heap, enum)
        else:  # gap
            true_node = enum_manager.true
            if node.is_true():
                prime, sub = true_node, true_node
            elif node.vtree.is_sub_vtree_of(vtree.left):
                prime, sub = node, true_node
            elif node.vtree.is_sub_vtree_of(vtree.right):
                prime, sub = true_node, node
            else:
                msg = "node vtree is not sub-vtree of given vtree"
                raise AssertionError(msg)
            enum = SddElementEnumerator(prime, sub, node, vtree, enum_manager)
            if not enum.empty(): heapq.heappush(self.heap, enum)

    def __iter__(self):
        return self

    def empty(self):
        return len(self.heap) == 0

    def __next__(self):
        while not self.empty():
            enum = heapq.heappop(self.heap)
            model = next(enum)
            self.topk.append(model)
            if not enum.empty(): heapq.heappush(self.heap, enum)
            return model
        raise StopIteration()

    def cached_iter(self):
        k = 0
        while True:
            if k < len(self.topk):
                yield self.topk[k]
                k += 1
            else:
                try:
                    next(self)
                except StopIteration:
                    return


class SddElementEnumerator:
    """Enumerator for SDD elements (prime/sub pairs)"""

    class HeapElement:
        def __init__(self, pinst, siter, element_enum, piter=None):
            self.pinst = pinst
            self.piter = piter
            self.siter = siter
            self.inst = None
            self.element_enum = element_enum
            self.enum_manager = element_enum.enum_manager

            self._try_next()

        def __iter__(self):
            return self

        def empty(self):
            return self.inst is None

        def _try_next(self):
            try:
                sinst = next(self.siter)
                self.inst = self.pinst.concat(sinst)
                self.enum_manager._element_update(self.element_enum, self.inst)
            except StopIteration:
                self.inst = None

        def __next__(self):
            if self.inst is None:
                raise StopIteration()
            else:
                saved_model = self.inst
                self._try_next()
                return saved_model

        def __cmp__(self, other):
            assert self.inst is not None and other.inst is not None
            return cmp(self.inst, other.inst)

        def __lt__(self, other):
            assert self.inst is not None and other.inst is not None
            return self.inst < other.inst

    def __init__(self, prime, sub, parent, vtree, enum_manager):
        self.prime = prime
        self.sub = sub
        self.parent = parent
        self.vtree = vtree
        self.enum_manager = enum_manager
        self.heap = []

        piter = enum_manager.lookup_node_enum(prime, vtree.left)
        self._push_next_element_enumerator(piter)

    def __iter__(self):
        return self

    def empty(self):
        return len(self.heap) == 0

    def _push_next_element_enumerator(self, piter):
        try:
            pinst = next(piter)
        except StopIteration:
            pinst = None
        if pinst is not None:
            siter = self.enum_manager.lookup_node_enum(self.sub, self.vtree.right)
            enum = SddElementEnumerator.HeapElement(pinst, siter, self, piter=piter)
            if not enum.empty(): heapq.heappush(self.heap, enum)

    def __next__(self):
        while not self.empty():
            best = heapq.heappop(self.heap)
            inst = next(best)

            if best.piter is not None:  # generate next prime model
                piter, best.piter = best.piter, None
                self._push_next_element_enumerator(piter)

            if not best.empty(): heapq.heappush(self.heap, best)
            return inst

        raise StopIteration()

    def __cmp__(self, other):
        assert not self.empty() and not other.empty()
        return cmp(self.heap[0].inst, other.heap[0].inst)

    def __lt__(self, other):
        assert not self.empty() and not other.empty()
        return self.heap[0].inst < other.heap[0].inst
