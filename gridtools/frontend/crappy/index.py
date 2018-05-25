import itertools
import math


class Index:
	_indices_count = itertools.count(0)

	def __init__(self, axis=None, name=None, offset=0):
		self._index_id = next(self._indices_count) if axis is None else axis
		self._name = name
		self._offset = offset

	def get_id(self):
		return self._index_id

	def get_name(self):
		return self._name

	def get_offset(self):
		return self._offset

	def get_indices_count(self):
		return self._indices_count

	def apply_offset(self, offset):
		self._offset += offset

	def __add__(self, other):
		"""
		Implement the addition between the current object and a scalar.
		Note: to accomodate for fractional offsets, the floor operator
		is applied to the scalar before the addition is performed.
		"""
		new_offset = self._offset + math.floor(other)
		return Index(axis=self._index_id, name=self._name, offset=new_offset)

	def __sub__(self, other):
		"""
		Implement the subtraction between the current object and a scalar.
		Note: to accomodate for fractional offsets, the ceil operator
		is applied to the scalar before the subtraction is performed.
		"""
		new_offset = self._offset - math.ceil(other)
		return Index(axis=self._index_id, name=self._name, offset=new_offset)


class IndicesTransformation:
	def __init__(self, source_indices, target_indices):
		source_indices, target_indices = self._make_inputs_iterable_if_necessary(source_indices, target_indices)
		self._source_indices = list(source_indices)
		self._target_indices = list(target_indices)
		self._rearranged_positions = self._compute_rearranged_positions(source_indices, target_indices)
		self._offsets_to_apply_to_source_indices = \
			self._compute_offsets_to_apply_to_source_indices(source_indices, target_indices)

	def get_source_indices(self):
		return self._source_indices

	def get_source_indices_ids(self):
		return [source_idx.get_id() for source_idx in self._source_indices]

	def get_target_indices(self):
		return self._target_indices

	def transform_indices(self, indices):
		import copy
		transformed_indices = [None] * len(indices)
		source_indices_used = [pos for pos in self._rearranged_positions if pos is not None]
		not_used_idx_position = len(source_indices_used)
		for source_position in range(0, len(indices)):
			if self._is_source_index_at_given_position_used(source_position):
				source_idx = indices[source_position]
				transformed_idx = copy.deepcopy(source_idx)
				transformed_idx.apply_offset(self.get_offset_to_apply_to_source_index(source_position))
				new_idx_position = self.get_rearranged_position_of_source_index(source_position)
				transformed_indices[new_idx_position] = transformed_idx
			else:
				source_idx = indices[source_position]
				transformed_idx = copy.deepcopy(source_idx)
				transformed_indices[not_used_idx_position] = transformed_idx
				not_used_idx_position += 1
		transformed_indices = [idx for idx in transformed_indices if idx is not None]
		return transformed_indices

	def add_source_index(self, source_index, position = None):
		if position is None:
			self._source_indices.append(source_index)
		else:
			self._source_indices.insert(position, source_index)
		self._rearranged_positions = self._compute_rearranged_positions(self._source_indices, self._target_indices)
		self._offsets_to_apply_to_source_indices = \
			self._compute_offsets_to_apply_to_source_indices(self._source_indices, self._target_indices)

	def _make_inputs_iterable_if_necessary(self, source_indices, target_indices):
		source_indices_out = [source_indices] if isinstance(source_indices, Index) else source_indices 
		target_indices_out = [target_indices] if isinstance(target_indices, Index) else target_indices 
		return source_indices_out, target_indices_out

	def _compute_rearranged_positions(self, source_indices, target_indices):
		rearranged_positions = []
		target_indices_rearranged_pos = dict( ((idx.get_id(), idx_pos) for idx_pos, idx in enumerate(target_indices)) )
		for source_idx in source_indices:
			source_idx_id = source_idx.get_id()
			rearranged_pos = target_indices_rearranged_pos.get(source_idx_id, None)
			rearranged_positions.append(rearranged_pos)
		return rearranged_positions

	def _compute_offsets_to_apply_to_source_indices(self, source_indices, target_indices):
		offsets = []
		target_indices_offsets = dict( ((idx.get_id(), idx.get_offset()) for idx in target_indices) )
		for source_idx in source_indices:
			source_idx_id = source_idx.get_id()
			if source_idx_id in target_indices_offsets:
				offset = target_indices_offsets[source_idx_id] - source_idx.get_offset()
			else:
				offset = 0
			offsets.append(offset)
		return offsets

	def _is_source_index_at_given_position_used(self, position):
		return self._rearranged_positions[position] is not None

	def get_rearranged_position_of_source_index(self, position_of_source_index):
		return self._rearranged_positions[position_of_source_index]

	def get_offset_to_apply_to_source_index(self, source_index_position):
		return self._offsets_to_apply_to_source_indices[source_index_position]

	def get_offsets_to_apply_to_source_indices(self):
		return self._offsets_to_apply_to_source_indices
