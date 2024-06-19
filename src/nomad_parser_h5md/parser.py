#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
import os
from typing import Any, Dict, List, Union

import h5py
import numpy as np
from ase.symbols import symbols2numbers
from atomisticparsers.utils import MOL, MDParser
from h5py import Group
from nomad.datamodel import EntryArchive
from nomad.metainfo.util import MEnum
from nomad.parsing.file_parser import FileParser
from nomad.units import ureg
from nomad_simulations import Program as BaseProgram

# New schema
from nomad_simulations import Simulation
from nomad_simulations.atoms_state import (
    AtomsState,
    CoreHole,
    HubbardInteractions,
    OrbitalsState,
)
from nomad_simulations.model_system import AtomicCell
from runschema.calculation import (
    BaseCalculation,
    Calculation,
    Energy,
    EnergyEntry,
)
from runschema.method import (
    AtomParameters,
    ForceCalculations,
    ForceField,
    Method,
    Model,
    NeighborSearching,
)
from runschema.run import MSection, Program, Run
from runschema.system import AtomsGroup, System
from simulationworkflowschema.molecular_dynamics import (
    CorrelationFunction,
    CorrelationFunctionValues,
    EnsembleProperty,
    EnsemblePropertyValues,
)

from .schema import Author, CalcEntry, ParamEntry
from .schema2 import (
    Author as Author2,
    ModelSystem,
    TrajectoryOutputs,
    OutputsEntry,
    TotalEnergy,
    TotalForce,
    ClassicalEnergyContributions,
    ForceContributions
)

class HDF5Parser(FileParser):
    def __init__(self):
        super().__init__(None)

    @property
    def filehdf5(self):
        if self._file_handler is None:
            try:
                self._file_handler = h5py.File(self.mainfile, 'r')
            except Exception:
                self.logger.error('Error reading hdf5 file.')

        return self._file_handler

    def apply_unit(self, quantity, unit: str, unit_factor: float):
        if quantity is None:
            return
        if unit:
            unit_val = ureg(unit)
            unit_val *= unit_factor
            quantity *= unit_val

        return quantity

    def decode_bytes(self, dataset):
        if dataset is None:
            return None
        elif isinstance(dataset, np.ndarray):
            if dataset.size == 0:
                return None
            dataset = (
                [val.decode('utf-8') for val in dataset]
                if isinstance(dataset[0], bytes)
                else dataset
            )
            dataset = (
                [val.__bool__() for val in dataset]
                if isinstance(dataset[0], bool)
                else dataset
            )
        elif (
            type(dataset).__name__ == 'bool_'
        ):  # TODO fix error when using isinstance() here
            dataset = dataset.__bool__()
        else:
            dataset = dataset.decode('utf-8') if isinstance(dataset, bytes) else dataset
        return dataset

    def get_attribute(
        self, group, attribute: str = None, path: str = None, default=None
    ):
        """
        Extracts attribute from group object based on path, and returns default if not defined.
        """
        if path:
            section_segments = path.split('.')
            for section in section_segments:
                try:
                    value = group.get(section)
                    group = value
                except Exception:
                    return
        value = group.attrs.get(attribute)
        value = self.decode_bytes(value) if value is not None else default

        return value if value is not None else default

    def get_value(self, group, path: str, default=None):
        """
        Extracts group or dataset from group object based on path, and returns default if not defined.
        """
        section_segments = path.split('.')
        for section in section_segments:
            try:
                value = group.get(section)
                unit = self.get_attribute(group, 'unit', path=section)
                unit_factor = self.get_attribute(
                    group, 'unit_factor', path=section, default=1.0
                )
                group = value
            except Exception:
                return

        if value is None:
            value = default
        elif isinstance(value, h5py.Dataset):
            value = value[()]
            value = self.apply_unit(value, unit, unit_factor)
        value = self.decode_bytes(value)

        return value if value is not None else default

    def parse(self, path: str = None, **kwargs):
        source = kwargs.get('source', self.filehdf5)
        isattr = kwargs.get('isattr', False)
        value = None
        if isattr:
            attr_path, attribute = path.rsplit('.', 1)
            value = self.get_attribute(source, attribute, path=attr_path)
        else:
            value = self.get_value(source, path)
        self._results[path] = value


class H5MDParser(MDParser):
    def __init__(self):
        super().__init__()
        self._data_parser = HDF5Parser()
        self._n_frames = None
        self._n_atoms = None
        self._atom_parameters = None
        self._system_info = None
        self._system_info2 = None
        self._observable_info = None
        self._parameter_info = None
        self._time_unit = ureg.picosecond
        self._path_group_particles_all = 'particles.all'
        self._path_group_positions_all = 'particles.all.position'
        self._path_value_positions_all = 'particles.all.position.value'

        self._nomad_to_particles_group_map = {
            'positions': 'position',
            'velocities': 'velocity',
            'forces': 'force',
            'labels': 'species_label',
            'label': 'force_field_label',
            'mass': 'mass',
            'charge': 'charge',
        }

        self._nomad_to_particles_group_map2 = {
            'positions': 'position',
            'velocities': 'velocity',
            'forces': 'force',
            'labels': 'species_label',
            'label': 'force_field_label',
            'mass': 'mass',
            'charge': 'charge',
        }

        self._nomad_to_box_group_map = {
            'lattice_vectors': 'edges',
            'periodic': 'boundary',
            'dimension': 'dimension',
        }

        self._nomad_to_box_group_map2 = {
            'lattice_vectors': 'edges',
            'periodic_boundary_conditions': 'boundary',
            'dimensionality': 'dimension',
        }

    def parse_atom_parameters(self):
        if self._n_atoms is None:
            return {}
        self._atom_parameters = {}
        n_atoms = self._n_atoms[0]  # TODO Extend to non-static n_atoms

        atom_parameter_keys = ['label', 'mass', 'charge']
        for key in atom_parameter_keys:
            value = self._data_parser.get(
                f'{self._path_group_particles_all}.{self._nomad_to_particles_group_map[key]}'
            )
            if value is not None:
                self._atom_parameters[key] = value
            else:
                continue
            if isinstance(self._atom_parameters[key], h5py.Group):
                self.logger.warning(
                    'Time-dependent atom parameters currently not supported.'
                    ' Atom parameter values will not be stored.'
                )
                continue
            elif len(self._atom_parameters[key]) != n_atoms:
                self.logger.warning(
                    'Inconsistent length of some atom parameters.'
                    ' Atom parameter values will not be stored.'
                )
                continue

    def parse_system_info(self):
        self._system_info = {'system': {}, 'calculation': {}}
        particles_group = self._data_parser.get(self._path_group_particles_all)
        positions = self._data_parser.get(self._path_value_positions_all)
        n_frames = self._n_frames
        if (
            particles_group is None or positions is None or positions is None
        ):  # For now we require that positions are present in the H5MD file to store other particle attributes
            self.logger.warning(
                'No positions available in H5MD file.'
                ' Other particle attributes will not be stored'
            )
            return self._system_info

        def get_value(value, steps, path=''):
            if value is None:
                return value
            if isinstance(value, h5py.Group):
                value = self._data_parser.get(f'{path}.value' if path else 'value')
                path_step = f'{path}.step' if path else 'step'
                attr_steps = self._data_parser.get(path_step)
                if value is None or attr_steps is None:
                    self.logger.warning(
                        'Missing values or steps in particle attributes.'
                        ' These attributes will not be stored.'
                    )
                    return None
                elif sorted(attr_steps) != sorted(steps):
                    self.logger.warning(
                        'Distinct trajectory lengths of particle attributes not supported.'
                        ' These attributes will not be stored.'
                    )
                    return None
                else:
                    return value
            else:
                return [value] * n_frames

        # get the steps based on the positions
        steps = self._data_parser.get(f'{self._path_group_positions_all}.step')
        if steps is None:
            self.logger.warning(
                'No step information available in H5MD file.'
                ' System information cannot be parsed.'
            )
            return self._system_info
        self.trajectory_steps = steps

        # get the rest of the particle quantities
        values_dict = {'system': {}, 'calculation': {}}
        times = self._data_parser.get(f'{self._path_group_positions_all}.time')
        values_dict['system']['time'] = times
        values_dict['calculation']['time'] = times
        values_dict['system']['positions'] = positions
        values_dict['system']['n_atoms'] = self._n_atoms
        system_keys = {
            'labels': 'system',
            'velocities': 'system',
            'forces': 'calculation',
        }
        for key, sec_key in system_keys.items():
            path = f'{self._path_group_particles_all}.{self._nomad_to_particles_group_map[key]}'
            value = self._data_parser.get(path)
            values_dict[sec_key][key] = get_value(value, steps, path=path)

        # get the box quantities
        box = self._data_parser.get(f'{self._path_group_particles_all}.box')
        if box is not None:
            box_attributes = ['dimension', 'periodic']
            for box_key in box_attributes:
                value = self._data_parser.get(
                    f'{self._path_group_particles_all}.box.{self._nomad_to_box_group_map[box_key]}',
                    isattr=True,
                )
                values_dict['system'][box_key] = (
                    [value] * n_frames if value is not None else None
                )

            box_key = 'lattice_vectors'
            path = f'{self._path_group_particles_all}.box.{self._nomad_to_box_group_map[box_key]}'
            value = self._data_parser.get(path)
            values_dict['system'][box_key] = get_value(value, steps, path=path)
        # populate the dictionary
        for i_step, step in enumerate(steps):
            self._system_info['system'][step] = {
                key: val[i_step]
                for key, val in values_dict['system'].items()
                if val is not None
            }
            self._system_info['calculation'][step] = {
                key: val[i_step]
                for key, val in values_dict['calculation'].items()
                if val is not None
            }

    def parse_system_info2(self):
        self._system_info2 = {'system': {}, 'calculation': {}}
        particles_group = self._data_parser.get(self._path_group_particles_all)
        positions = self._data_parser.get(self._path_value_positions_all)
        n_frames = self._n_frames
        if (
            particles_group is None or positions is None or positions is None
        ):  # For now we require that positions are present in the H5MD file to store other particle attributes
            self.logger.warning(
                'No positions available in H5MD file.'
                ' Other particle attributes will not be stored'
            )
            return self._system_info2

        def get_value(value, steps, path=''):
            if value is None:
                return value
            if isinstance(value, h5py.Group):
                value = self._data_parser.get(f'{path}.value' if path else 'value')
                path_step = f'{path}.step' if path else 'step'
                attr_steps = self._data_parser.get(path_step)
                if value is None or attr_steps is None:
                    self.logger.warning(
                        'Missing values or steps in particle attributes.'
                        ' These attributes will not be stored.'
                    )
                    return None
                elif sorted(attr_steps) != sorted(steps):
                    self.logger.warning(
                        'Distinct trajectory lengths of particle attributes not supported.'
                        ' These attributes will not be stored.'
                    )
                    return None
                else:
                    return value
            else:
                return [value] * n_frames

        # get the steps based on the positions
        steps = self._data_parser.get(f'{self._path_group_positions_all}.step')
        if steps is None:
            self.logger.warning(
                'No step information available in H5MD file.'
                ' System information cannot be parsed.'
            )
            return self._system_info2
        self.trajectory_steps = steps

        # get the rest of the particle quantities
        values_dict = {'system': {}, 'calculation': {}}
        times = self._data_parser.get(f'{self._path_group_positions_all}.time')
        values_dict['system']['time'] = times
        values_dict['calculation']['time'] = times
        values_dict['system']['positions'] = positions
        values_dict['system']['n_atoms'] = self._n_atoms
        system_keys = {
            'labels': 'system',
            'velocities': 'system',
            'forces': 'calculation',
        }
        for key, sec_key in system_keys.items():
            path = f'{self._path_group_particles_all}.{self._nomad_to_particles_group_map2[key]}'
            value = self._data_parser.get(path)
            values_dict[sec_key][key] = get_value(value, steps, path=path)

        # get the box quantities
        box = self._data_parser.get(f'{self._path_group_particles_all}.box')
        if box is not None:
            box_attributes = ['dimensionality', 'periodic_boundary_conditions']
            for box_key in box_attributes:
                value = self._data_parser.get(
                    f'{self._path_group_particles_all}.box.{self._nomad_to_box_group_map2[box_key]}',
                    isattr=True,
                )
                values_dict['system'][box_key] = (
                    [value] * n_frames if value is not None else None
                )

            box_key = 'lattice_vectors'
            path = f'{self._path_group_particles_all}.box.{self._nomad_to_box_group_map2[box_key]}'
            value = self._data_parser.get(path)
            values_dict['system'][box_key] = get_value(value, steps, path=path)
        # populate the dictionary
        for i_step, step in enumerate(steps):
            self._system_info2['system'][step] = {
                key: val[i_step]
                for key, val in values_dict['system'].items()
                if val is not None
            }
            self._system_info2['calculation'][step] = {
                key: val[i_step]
                for key, val in values_dict['calculation'].items()
                if val is not None
            }

    def parse_observable_info(self):
        self._observable_info = {
            'configurational': {},
            'ensemble_average': {},
            'correlation_function': {},
        }
        thermodynamics_steps = []
        observables_group = self._data_parser.get('observables')
        if observables_group is None:
            return self._observable_info

        def get_observable_paths(observable_group: Dict, current_path: str) -> List:
            paths = []
            for obs_key in observable_group.keys():
                path = f'{obs_key}.'
                observable = self._data_parser.get_value(observable_group, obs_key)
                observable_type = self._data_parser.get_value(
                    observable_group, obs_key
                ).attrs.get('type')
                if not observable_type:
                    paths.extend(
                        get_observable_paths(observable, f'{current_path}{path}')
                    )
                else:
                    paths.append(current_path + path[:-1])

            return paths

        observable_paths = get_observable_paths(observables_group, current_path='')
        for path in observable_paths:
            full_path = f'observables.{path}'
            observable = self._data_parser.get_value(observables_group, path)
            observable_type = self._data_parser.get_value(
                observables_group, path
            ).attrs.get('type')
            observable_name, observable_label = (
                path.split('.', 1) if len(path.split('.')) > 1 else [path, '']
            )
            if observable_type == 'configurational':
                steps = self._data_parser.get(f'{full_path}.step')
                if steps is None:
                    self.logger.warning(
                        'Missing step information in some observables.'
                        ' These will not be stored.'
                    )
                    continue
                thermodynamics_steps = set(list(steps) + list(thermodynamics_steps))
                times = self._data_parser.get(f'{full_path}.time')
                values = self._data_parser.get(f'{full_path}.value')
                if isinstance(values, h5py.Group):
                    self.logger.warning(
                        'Group structures within individual observables not supported.'
                        ' These will not be stored.'
                    )
                    continue
                for i_step, step in enumerate(steps):
                    if not self._observable_info[observable_type].get(step):
                        self._observable_info[observable_type][step] = {}
                        self._observable_info[observable_type][step]['time'] = times[
                            i_step
                        ]
                    observable_key = (
                        f'{observable_name}-{observable_label}'
                        if observable_label
                        else f'{observable_name}'
                    )
                    self._observable_info[observable_type][step][observable_key] = (
                        values[i_step]
                    )
            else:
                if observable_name not in self._observable_info[observable_type].keys():
                    self._observable_info[observable_type][observable_name] = {}
                self._observable_info[observable_type][observable_name][
                    observable_label
                ] = {}
                for key in observable.keys():
                    observable_attribute = self._data_parser.get(f'{full_path}.{key}')
                    if isinstance(observable_attribute, h5py.Group):
                        self.logger.warning(
                            'Group structures within individual observables not supported.'
                            ' These will not be stored.'
                        )
                        continue
                    self._observable_info[observable_type][observable_name][
                        observable_label
                    ][key] = observable_attribute

            self.thermodynamics_steps = thermodynamics_steps

    def parse_atomsgroup(
        self,
        nomad_sec: Union[System, AtomsGroup],
        h5md_sec_particlesgroup: Group,
        path_particlesgroup: str,
    ):
        for i_key, key in enumerate(h5md_sec_particlesgroup.keys()):
            path_particlesgroup_key = f'{path_particlesgroup}.{key}'
            particles_group = {
                group_key: self._data_parser.get(
                    f'{path_particlesgroup_key}.{group_key}'
                )
                for group_key in h5md_sec_particlesgroup[key].keys()
            }
            sec_atomsgroup = AtomsGroup()
            nomad_sec.atoms_group.append(sec_atomsgroup)
            sec_atomsgroup.type = particles_group.pop('type', None)
            sec_atomsgroup.index = i_key
            sec_atomsgroup.atom_indices = particles_group.pop('indices', None)
            sec_atomsgroup.n_atoms = (
                len(sec_atomsgroup.atom_indices)
                if sec_atomsgroup.atom_indices is not None
                else None
            )
            sec_atomsgroup.is_molecule = particles_group.pop('is_molecule', None)
            sec_atomsgroup.label = particles_group.pop('label', None)
            sec_atomsgroup.composition_formula = particles_group.pop('formula', None)
            particles_subgroup = particles_group.pop('particles_group', None)
            # set the remaining attributes
            for particles_group_key in particles_group.keys():
                val = particles_group.get(particles_group_key)
                units = val.units if hasattr(val, 'units') else None
                val = val.magnitude if units is not None else val
                sec_atomsgroup.x_h5md_parameters.append(
                    ParamEntry(kind=particles_group_key, value=val, unit=units)
                )
            # get the next atomsgroup
            if particles_subgroup:
                self.parse_atomsgroup(
                    sec_atomsgroup,
                    particles_subgroup,
                    f'{path_particlesgroup_key}.particles_group',
                )

    def is_valid_key_val(self, metainfo_class: MSection, key: str, val) -> bool:
        if hasattr(metainfo_class, key):
            quant_type = getattr(metainfo_class, key).get('type')
            is_menum = isinstance(quant_type, MEnum) if quant_type else False
            return False if is_menum and val not in quant_type._list else True
        else:
            return False

    def parse_parameter_info(self):
        self._parameter_info = {'force_calculations': {}, 'workflow': {}}

        def get_parameters(parameter_group: Group, path: str) -> Dict:
            param_dict: Dict[Any, Any] = {}
            for key, val in parameter_group.items():
                path_key = f'{path}.{key}'
                if isinstance(val, h5py.Group):
                    param_dict[key] = get_parameters(val, path_key)
                else:
                    param_dict[key] = self._data_parser.get(path_key)
                    if isinstance(param_dict[key], str):
                        param_dict[key] = (
                            param_dict[key].upper()
                            if key == 'thermodynamic_ensemble'
                            else param_dict[key].lower()
                        )
                    elif isinstance(param_dict[key], (int, np.int32, np.int64)):
                        param_dict[key] = param_dict[key].item()

            return param_dict

        force_calculations_group = self._data_parser.get(
            'parameters.force_calculations'
        )
        if force_calculations_group is not None:
            self._parameter_info['force_calculations'] = get_parameters(
                force_calculations_group, 'parameters.force_calculations'
            )
        workflow_group = self._data_parser.get('parameters.workflow')
        if workflow_group is not None:
            self._parameter_info['workflow'] = get_parameters(
                workflow_group, 'parameters.workflow'
            )

    def parse_calculation(self):
        sec_run = self.archive.run[-1]
        calculation_info = self._observable_info.get('configurational')
        if (
            not calculation_info
        ):  # TODO should still create entries for system time link in this case
            return

        system_info = self._system_info.get(
            'calculation'
        )  # note: it is currently ensured in parse_system() that these have the same length as the system_map
        for step in self.steps:
            data = {
                'method_ref': sec_run.method[-1] if sec_run.method else None,
                'step': step,
                'energy': {},
            }
            data_h5md = {
                'x_h5md_custom_calculations': [],
                'x_h5md_energy_contributions': [],
            }
            #! Apparently this is not being set here, but where?!
            # data['time'] = calculation_info.get(step, {}).get('time')
            # if not data['time']:
            #     data['time'] = system_info.get(step, {}).get('time')

            for key, val in system_info.get(step, {}).items():
                if key == 'forces':
                    data[key] = dict(total=dict(value=val))
                elif hasattr(BaseCalculation, key):
                    data[key] = val
                else:
                    unit = None
                    if hasattr(val, 'units'):
                        unit = val.units
                        val = val.magnitude
                    data_h5md['x_h5md_custom_calculations'].append(
                        CalcEntry(kind=key, value=val, unit=unit)
                    )

            for key, val in calculation_info.get(step).items():
                key_split = key.split('-')
                observable_name = key_split[0]
                observable_label = key_split[1] if len(key_split) > 1 else key_split[0]
                if (
                    'energ' in observable_name
                ):  # TODO check for energies or energy when matching name
                    # check for usage of energy/mole and convert to energy
                    if val.check("[energy]/[substance]") and "mole" in str(val.units):
                        val = val * MOL * ureg.mole

                    if val.check("[energy]"):
                        if hasattr(Energy, observable_label):
                            data["energy"][observable_label] = dict(value=val)
                        else:
                            data_h5md["x_h5md_energy_contributions"].append(
                                EnergyEntry(kind=key, value=val)
                            )
                    else:
                        self.logger.warning(
                            "Energy value not in energy units. Skipping entry."
                        )
                elif hasattr(BaseCalculation, observable_label):
                    data[observable_label] = val
                else:
                    unit = None
                    if hasattr(val, 'units'):
                        unit = val.units
                        val = val.magnitude
                    data_h5md['x_h5md_custom_calculations'].append(
                        CalcEntry(kind=key, value=val, unit=unit)
                    )

            self.parse_thermodynamics_step(data)
            sec_calc = sec_run.calculation[-1]

            if sec_calc.step != step:  # TODO check this comparison
                sec_calc = Calculation()
                sec_run.calculation.append(sec_calc)
                sec_calc.step = int(step)
                sec_calc.time = data['time']
            for calc_entry in data_h5md['x_h5md_custom_calculations']:
                sec_calc.x_h5md_custom_calculations.append(calc_entry)
            sec_energy = sec_calc.energy
            if not sec_energy:
                sec_energy = Energy()
                sec_calc.append(sec_energy)
            for energy_entry in data_h5md['x_h5md_energy_contributions']:
                sec_energy.x_h5md_energy_contributions.append(energy_entry)

    def parse_system(self):
        sec_run = self.archive.run[-1]

        system_info = self._system_info.get('system')
        if not system_info:
            self.logger.error('No particle information found in H5MD file.')
            return

        self._system_time_map = {}
        for i_step, step in enumerate(self.trajectory_steps):
            # time = system_info[step].pop("time") # unused!
            atoms_dict = system_info[step]

            atom_labels = atoms_dict.get('labels')
            if atom_labels is not None:
                try:
                    symbols2numbers(atom_labels)
                except KeyError:  # TODO this check should be moved to the system normalizer in the new schema
                    atoms_dict['labels'] = ['X'] * len(atom_labels)

            topology = None
            if i_step == 0:  # TODO extend to time-dependent bond lists and topologies
                atoms_dict['bond_list'] = self._data_parser.get('connectivity.bonds')
                path_topology = 'connectivity.particles_group'
                topology = self._data_parser.get(path_topology)

            self.parse_trajectory_step({'atoms': atoms_dict})

            if i_step == 0 and topology:  # TODO extend to time-dependent topologies
                self.parse_atomsgroup(sec_run.system[i_step], topology, path_topology)

    def parse_method(self):
        sec_method = Method()
        self.archive.run[-1].method.append(sec_method)
        sec_force_field = ForceField()
        sec_method.force_field = sec_force_field
        sec_model = Model()
        sec_force_field.model.append(sec_model)

        # get the atom parameters
        n_atoms = (
            self._n_atoms[0] if self._n_atoms is not None else 0
        )  # TODO Extend to non-static n_atoms
        for n in range(n_atoms):
            sec_atom = AtomParameters()
            sec_method.atom_parameters.append(sec_atom)

            for key in self._atom_parameters.keys():
                sec_atom.m_set(
                    sec_atom.m_get_quantity_definition(key),
                    self._atom_parameters[key][n],
                )

        # Get the interactions
        connectivity_group = self._data_parser.get('connectivity')
        if connectivity_group:
            atom_labels = self._atom_parameters.get('label')
            interaction_keys = ['bonds', 'angles', 'dihedrals', 'impropers']
            interactions_by_type = []
            for interaction_key in interaction_keys:
                interaction_list = self._data_parser.get(
                    f'connectivity.{interaction_key}'
                )
                if interaction_list is None:
                    continue
                elif isinstance(interaction_list, h5py.Group):
                    self.logger.warning(
                        'Time-dependent interactions currently not supported.'
                        ' These values will not be stored'
                    )
                    continue

                interaction_type_dict = {
                    'type': interaction_key,
                    'n_interactions': len(interaction_list),
                    'n_atoms': len(interaction_list[0]),
                    'atom_indices': interaction_list,
                    'atom_labels': np.array(atom_labels)[interaction_list]
                    if atom_labels is not None
                    else None,
                }
                interactions_by_type.append(interaction_type_dict)
            self.parse_interactions_by_type(interactions_by_type, sec_model)

        # Get the force calculation parameters
        force_calculation_parameters = self._parameter_info.get('force_calculations')
        if force_calculation_parameters:
            sec_force_calculations = ForceCalculations()
            sec_force_field.force_calculations = sec_force_calculations
            sec_neighbor_searching = NeighborSearching()
            sec_force_calculations.neighbor_searching = sec_neighbor_searching

            for key, val in force_calculation_parameters.items():
                if not isinstance(val, dict):
                    if self.is_valid_key_val(ForceCalculations, key, val):
                        sec_force_calculations.m_set(
                            sec_force_calculations.m_get_quantity_definition(key), val
                        )
                    else:
                        units = val.units if hasattr(val, 'units') else None
                        val = val.magnitude if units is not None else val
                        sec_force_calculations.x_h5md_parameters.append(
                            ParamEntry(kind=key, value=val, unit=units)
                        )
                elif key == 'neighbor_searching':
                    for neigh_key, neigh_val in val.items():
                        if self.is_valid_key_val(
                            NeighborSearching, neigh_key, neigh_val
                        ):
                            sec_neighbor_searching.m_set(
                                sec_neighbor_searching.m_get_quantity_definition(
                                    neigh_key
                                ),
                                neigh_val,
                            )
                        else:
                            units = val.units if hasattr(val, 'units') else None
                            val = val.magnitude if units is not None else val
                            sec_neighbor_searching.x_h5md_parameters.append(
                                ParamEntry(kind=key, value=val, unit=units)
                            )
                else:
                    self.logger.warning(
                        'Unknown parameters in force calculations section.'
                        ' These will not be stored.'
                    )

    def get_workflow_properties_dict(
        self,
        observables: Dict,
        property_type_key=None,
        property_type_value_key=None,
        properties_known={},
        property_keys_list=[],
        property_value_keys_list=[],
    ):
        def populate_property_dict(
            property_dict, val_name, val, flag_known_property=False
        ):
            if val is None:
                return
            value_unit = val.units if hasattr(val, 'units') else None
            value_magnitude = val.magnitude if hasattr(val, 'units') else val
            if flag_known_property:
                property_dict[val_name] = (
                    value_magnitude * value_unit if value_unit else value_magnitude
                )
            else:
                property_dict[f'{val_name}_unit'] = (
                    str(value_unit) if value_unit else None
                )
                property_dict[f'{val_name}_magnitude'] = value_magnitude

        workflow_properties_dict: Dict[Any, Any] = {}
        for observable_type, observable_dict in observables.items():
            flag_known_property = False
            if observable_type in properties_known:
                property_type_key = observable_type
                property_type_value_key = properties_known[observable_type]
                flag_known_property = True
            property_dict: Dict[Any, Any] = {property_type_value_key: []}
            property_dict['label'] = observable_type
            for key, observable in observable_dict.items():
                property_values_dict = {'label': key}
                for quant_name, val in observable.items():
                    if quant_name == 'val':
                        continue
                    if quant_name == 'bins':
                        continue
                    if quant_name in property_keys_list:
                        property_dict[quant_name] = val
                    if quant_name in property_value_keys_list:
                        property_values_dict[quant_name] = val
                    # TODO Still need to add custom values here.

                val = observable.get('value')
                populate_property_dict(
                    property_values_dict,
                    'value',
                    val,
                    flag_known_property=flag_known_property,
                )
                bins = observable.get('bins')
                populate_property_dict(
                    property_values_dict,
                    'bins',
                    bins,
                    flag_known_property=flag_known_property,
                )
                property_dict[property_type_value_key].append(property_values_dict)

            if workflow_properties_dict.get(property_type_key):
                workflow_properties_dict[property_type_key].append(property_dict)
            else:
                workflow_properties_dict[property_type_key] = [property_dict]

        return workflow_properties_dict

    def parse_workflow(self):
        workflow_parameters = self._parameter_info.get('workflow').get(
            'molecular_dynamics'
        )
        if workflow_parameters is None:
            return

        workflow_results = {}
        ensemble_average_observables = self._observable_info.get('ensemble_average')
        ensemble_property_dict = {
            'property_type_key': 'ensemble_properties',
            'property_type_value_key': 'ensemble_property_values',
            'properties_known': {
                'radial_distribution_functions': 'radial_distribution_function_values'
            },
            'property_keys_list': EnsembleProperty.m_def.all_quantities.keys(),
            'property_value_keys_list': EnsemblePropertyValues.m_def.all_quantities.keys(),
        }
        workflow_results.update(
            self.get_workflow_properties_dict(
                ensemble_average_observables, **ensemble_property_dict
            )
        )
        correlation_function_observables = self._observable_info.get(
            'correlation_function'
        )
        correlation_function_dict = {
            'property_type_key': 'correlation_functions',
            'property_type_value_key': 'correlation_function_values',
            'properties_known': {
                'mean_squared_displacements': 'mean_squared_displacement_values'
            },
            'property_keys_list': CorrelationFunction.m_def.all_quantities.keys(),
            'property_value_keys_list': CorrelationFunctionValues.m_def.all_quantities.keys(),
        }
        workflow_results.update(
            self.get_workflow_properties_dict(
                correlation_function_observables, **correlation_function_dict
            )
        )
        self.parse_md_workflow(
            dict(method=workflow_parameters, results=workflow_results)
        )

    def parse_h5md_group(self) -> dict:
        group_h5md = self._data_parser.get('h5md')
        group_h5md_dict = {}
        if group_h5md:
            group_h5md_dict['program_name'] = self._data_parser.get(
                'h5md.program.name', isattr=True
            )
            group_h5md_dict['program_version'] = self._data_parser.get(
                'h5md.program.version', isattr=True
            )
            group_h5md_dict['h5md_version'] = self._data_parser.get(
                'h5md.version', isattr=True
            )
            group_h5md_dict['h5md_author_name'] = self._data_parser.get(
                'h5md.author.name', isattr=True
            )
            group_h5md_dict['h5md_author_email'] = self._data_parser.get(
                'h5md.author.email', isattr=True
            )
            group_h5md_dict['h5md_creator_name'] = self._data_parser.get(
                'h5md.creator.name', isattr=True
            )
            group_h5md_dict['h5md_creator_version'] = self._data_parser.get(
                'h5md.creator.version', isattr=True
            )
        else:
            self.logger.warning(
                '"h5md" group missing in (H5MD)hdf5 file.'
                ' Program and author metadata will be missing!'
            )

        return group_h5md_dict

    def parse_system_hierarchy(
        self,
        nomad_sec: ModelSystem,
        h5md_sec_particlesgroup: Group,
        path_particlesgroup: str,
    ):
        data = {}
        for key in h5md_sec_particlesgroup.keys():
            path_particlesgroup_key = f'{path_particlesgroup}.{key}'
            particles_group = {
                group_key: self._data_parser.get(
                    f'{path_particlesgroup_key}.{group_key}'
                )
                for group_key in h5md_sec_particlesgroup[key].keys()
            }
            sec_model_system = ModelSystem()
            nomad_sec.model_system.append(sec_model_system)
            data['branch_label'] = particles_group.pop('label', None)
            data['atom_indices'] = particles_group.pop('indices', None)
            # TODO remove the deprecated below from the test file
            # sec_atomsgroup.type = particles_group.pop("type", None) #? deprecate?
            particles_group.pop('type', None)
            # sec_atomsgroup.is_molecule = particles_group.pop("is_molecule", None) #? deprecate?
            particles_group.pop('is_molecule', None)
            particles_group.pop('formula', None)  # covered in normalization now
            # write all the standard quantities to the archive
            self.parse_section(data, sec_model_system)
            particles_subgroup = particles_group.pop('particles_group', None)

            # set the remaining attributes
            for particles_group_key in particles_group.keys():
                val = particles_group.get(particles_group_key)
                units = val.units if hasattr(val, 'units') else None
                val = val.magnitude if units is not None else val
                sec_model_system.custom_system_attributes.append(
                    ParamEntry(kind=particles_group_key, value=val, unit=units)
                )

            # get the next branch level
            if particles_subgroup:
                self.parse_system_hierarchy(
                    sec_model_system,
                    particles_subgroup,
                    f'{path_particlesgroup_key}.particles_group',
                )

    # TODO move this function to the MDParser class
    def parse_trajectory_step2(
        self, data: Dict[str, Any], simulation: Simulation
    ) -> None:
        """
        Create a system section and write the provided data.
        """
        if self.archive is None:
            return

        if (step := data.get('step')) is not None and step not in self.trajectory_steps:
            return

        model_system = ModelSystem()
        atomic_cell = AtomicCell()
        atomic_cell_dict = data.pop('atomic_cell')
        atom_labels = atomic_cell_dict.pop('labels')
        for label in atom_labels:
            atoms_state = AtomsState(chemical_symbol=label)
            atomic_cell.atoms_state.append(atoms_state)
        self.parse_section(atomic_cell_dict, atomic_cell)
        model_system.cell.append(atomic_cell)
        self.parse_section(data, model_system)
        simulation.model_system.append(model_system)

        return model_system

    def parse_system2(self, simulation):
        system_info = self._system_info2.get('system')
        if not system_info:
            self.logger.error('No particle information found in H5MD file.')
            return

        self._system_time_map = {}
        for i_step, step in enumerate(self.trajectory_steps):
            atoms_dict = system_info[step]
            atoms_dict['is_representative'] = False

            atom_labels = atoms_dict.get('labels')
            if atom_labels is not None:
                try:
                    symbols2numbers(atom_labels)
                    atoms_dict['labels'] = atom_labels
                except KeyError:  # TODO this check should be moved to the system normalizer in the new schema
                    atoms_dict['labels'] = ['X'] * len(atom_labels)

            topology = None
            if i_step == 0:  # TODO extend to time-dependent bond lists and topologies
                atoms_dict['is_representative'] = True
                atoms_dict['bond_list'] = self._data_parser.get('connectivity.bonds')
                path_topology = 'connectivity.particles_group'
                topology = self._data_parser.get(path_topology)

            # REMAP some of the data for the schema
            atoms_dict['branch_label'] = (
                'Total System'  # ? Do we or should we have a default name for the entire system?
            )
            atoms_dict['time_step'] = atoms_dict.pop(
                'time'
            ).magnitude  # TODO change in system_info
            atomic_cell_keys = [
                'n_atoms',
                'lattice_vectors',
                'periodic_boundary_conditions',
                'positions',
                'velocities',
                'labels',
            ]
            atoms_dict['atomic_cell'] = {}
            for key in atomic_cell_keys:
                atoms_dict['atomic_cell'][key] = atoms_dict.pop(key)

            self.parse_trajectory_step2(atoms_dict, simulation)

            if i_step == 0 and topology:  # TODO extend to time-dependent topologies
                self.parse_system_hierarchy(
                    simulation.model_system[-1], topology, path_topology
                )

    # def parse_section_test(self, data: Dict[str, Any], root: MSection) -> None:
    #     """
    #     Write the quantities in data into an archive section.
    #     """
    #     from nomad.metainfo import SubSection
    #     print(data.items())
    #     for key, val in data.items():
    #         if not hasattr(root, key):
    #             continue

    #         print(key, val)
    #         print(root)
    #         print(root.m_def)
    #         section = getattr(root.m_def.section_cls, key)
    #         print(section)
    #         print(section.m_def)
    #         if isinstance(section, SubSection):
    #             print(val)
    #             print(isinstance(val, dict))
    #             print([val])
    #             print([val_n for val_n in [val]])
    #             for val_n in [val] if isinstance(val, dict) else val:
    #                 print(val_n)
    #                 print(section.sub_section.section_cls())
    #                 sub_section = section.sub_section.section_cls()
    #                 root.m_add_sub_section(section, sub_section)
    #                 print(root)
    #                 self.parse_section_test(val_n, sub_section)
    #             continue

    #         root.m_set(root.m_get_quantity_definition(key), val)

    # TODO move this function to the MDParser class and rename!
    def parse_output_step(self, data: Dict[str, Any], simulation: Simulation) -> bool:
        if self.archive is None:
            return False

        if (
            step := data.get("step")
        ) is not None and step not in self.thermodynamics_steps:
            return False

        output = TrajectoryOutputs() # Outputs(model_system_ref=simulation.model_system[-1])
        simulation.outputs.append(output)

        # output.step = data['step']
        # output.time = data['time']
        # from nomad_simulations.properties import TotalEnergy
        # total_energy = TotalEnergy(value = data['total_energy']['value'])
        # output.total_energy.append(total_energy)
        # classical_contibutions = ClassicalEnergyContributions()
        # total_energy.classical_contributions = classical_contibutions
        # classical_contibutions.kinetic = data['total_energy']['classical_contributions']['kinetic']
        # classical_contibutions.potential = data['total_energy']['classical_contributions']['potential']
        self.parse_section(data, output)
        try:
            system_ref_index = self.trajectory_steps.index(output.step)
            output.model_system_ref = simulation.model_system[system_ref_index]
        except Exception:
            self.logger.warning('Could not set system reference in parsing of outputs.')

        return True

    def parse_outputs(self, simulation: Simulation):
        outputs_info = self._observable_info.get('configurational')
        if (
            not outputs_info
        ):  # TODO should still create entries for system time link in this case
            return

        system_info = self._system_info2.get(
            'calculation'
        )  # note: it is currently ensured in parse_system() that these have the same length as the system_map

        for step in self.steps:
            data = {
                # 'method_ref': simulation.method[-1] if simulation.method else None,
                'step': step,
                'total_energy': {'classical_contributions': {}},
                'total_force': {'contributions': {}},
            } # nb - only allowing 1 contribution to total energy and total force for now
            data_h5md = {
                'x_h5md_custom_calculations': [],
                'x_h5md_energy_contributions': [],
                'x_h5md_force_contributions': [],
            }
            data['time'] = outputs_info.get(step, {}).get('time')
            if not data['time']:
                data['time'] = system_info.get(step, {}).get('time')

            # TODO decide if forces will stay in system_info and will be placed still in outputs
            forces = system_info.get(step, {}).get('forces')
            if forces is not None:
                data['total_force']['value'] = forces
            # for key, val in system_info.get(step, {}).items():
            #     if key == 'forces':
            #         data['total_force']['value'] = val
            #     elif hasattr(TrajectoryOutputs, key):
            #         data[key]['value'] = val
            #     else:
            #         unit = None
            #         if hasattr(val, 'units'):
            #             unit = val.units
            #             val = val.magnitude
            #         data_h5md['x_h5md_custom_calculations'].append(
            #             OutputsEntry(kind=key, value=val, unit=unit)
            #         )

            for key, val in outputs_info.get(step).items():
                key_split = key.split('-')
                observable_name = key_split[0]
                observable_label = key_split[1] if len(key_split) > 1 else key_split[0]
                if observable_name == 'total_energy':
                    data['total_energy']['value'] = val
                elif (
                    'energ' in observable_name
                ):  # TODO check for energies or energy when matching name
                    # TODO add support for energy/mole as in parse_calculation
                    if hasattr(ClassicalEnergyContributions, observable_label):
                        data['total_energy']['classical_contributions'][observable_label] = {'value': val}
                    else:
                        data_h5md['x_h5md_energy_contributions'].append(
                            OutputsEntry(name=key, value=val.magnitude, unit=val.units)
                        )
                # elif observable_name == 'forces':
                #     data['total_force']['value'] = val
                elif (
                    'force' in observable_name
                ):
                    if hasattr(ForceContributions, observable_label):
                        data['total_force']['contributions'][observable_label] = {'value': val}
                    else:
                        data_h5md['x_h5md_force_contributions'].append(
                            OutputsEntry(name=key, value=val.magnitude, unit=val.units)
                        )
                elif hasattr(TrajectoryOutputs, observable_label):
                    data[observable_label] = val
                else:
                    unit = None
                    if hasattr(val, 'units'):
                        unit = val.units
                        val = val.magnitude
                    data_h5md['x_h5md_custom_calculations'].append(
                        OutputsEntry(name=key, value=val, unit=unit)
                    )

            flag_parsed = self.parse_output_step(data, simulation)
            if flag_parsed:
                output = simulation.outputs[-1]
                for output_entry in data_h5md['x_h5md_custom_calculations']:
                    output.x_h5md_custom_outputs.append(output_entry)
                if len(output.total_energy) == 0 and data_h5md['x_h5md_energy_contributions']:
                    total_energy = TotalEnergy()
                    output.total_energy.append(total_energy)
                sec_contributions = output.total_energy[0].classical_contributions
                for energy_entry in data_h5md['x_h5md_energy_contributions']:
                    sec_contributions.x_h5md_contributions.append(energy_entry)
                if len(output.total_force) == 0 and data_h5md['x_h5md_force_contributions']:
                    total_force = TotalForce()
                    output.total_force.append(total_force)
                sec_contributions = output.total_force[0].contributions
                for force_entry in data_h5md['x_h5md_force_contributions']:
                    sec_contributions.x_h5md_contributions.append(force_entry)

    def write_to_archive(self) -> None:
        self._maindir = os.path.dirname(self.mainfile)
        self._h5md_files = os.listdir(self._maindir)
        self._basename = os.path.basename(self.mainfile).rsplit('.', 1)[0]
        self._data_parser.mainfile = self.mainfile
        if self._data_parser.filehdf5 is None:
            self.logger.warning('hdf5 file missing in H5MD Parser.')
            return

        positions = self._data_parser.get(self._path_value_positions_all)
        if positions is not None:
            self._n_frames = len(positions) if positions is not None else None
            self._n_atoms = (
                [len(pos) for pos in positions] if positions is not None else None
            )
        # Parse the hdf5 groups
        group_h5md_dict = self.parse_h5md_group()

        self.parse_atom_parameters()
        self.parse_system_info()
        self.parse_system_info2()
        self.parse_observable_info()
        # self.parse_parameter_info()

        ###########################
        # Populate the OLD SCHEMA #
        ###########################
        sec_run = Run()
        self.archive.run.append(sec_run)
        sec_run.program = Program(
            name=group_h5md_dict.get('program_name'),
            version=group_h5md_dict.get('program_version'),
        )
        sec_run.x_h5md_version = group_h5md_dict.get('h5md_version')
        sec_run.x_h5md_author = Author(
            name=group_h5md_dict.get('h5md_author_name'),
            email=group_h5md_dict.get('h5md_author_email'),
        )
        sec_run.x_h5md_creator = Program(
            name=group_h5md_dict.get('h5md_creator_name'),
            version=group_h5md_dict.get('h5md_creator_version'),
        )

        # self.parse_method()

        self.parse_system()

        self.parse_calculation()

        # self.parse_workflow()

        ###########################
        # Populate the NEW SCHEMA #
        ###########################
        simulation = Simulation()
        simulation.program = BaseProgram(
            name=group_h5md_dict.get('program_name'),
            version=group_h5md_dict.get('program_version'),
        )
        simulation.x_h5md_version = group_h5md_dict.get('h5md_version')
        simulation.x_h5md_author = Author2(
            name=group_h5md_dict.get('h5md_author_name'),
            email=group_h5md_dict.get('h5md_author_email'),
        )
        simulation.x_h5md_creator = BaseProgram(
            name=group_h5md_dict.get('h5md_creator_name'),
            version=group_h5md_dict.get('h5md_creator_version'),
        )

        self.parse_system2(simulation)

        # # self.parse_method(simulation)

        self.parse_outputs(simulation)

        # self.parse_workflow(simulation)

        self.archive.m_add_sub_section(EntryArchive.data, simulation)
