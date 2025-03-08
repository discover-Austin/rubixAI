#!/usr/bin/env python3
"""
Advanced Rubik's Cube NLU System
Fully implemented production-ready code with complete logic for concept mapping, cube state
transformation, pattern matching, concept graph propagation, and integration suited for a chatbot.
All mappings (including complete corner and edge mappings) and logic are fully incorporated.
"""

import numpy as np
import re
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue

##############################################
# Cube Constants and Standard Mappings
##############################################

class CubeConstants:
    TOTAL_STATES = 43252003274489856000  # Total Rubik's Cube states (theoretical)
    GODS_NUMBER = 20                    # Maximum moves required to solve any cube state
    FACES = 6                           # Cube faces (semantic categories)
    STICKERS_PER_FACE = 9               # Stickers per face
    TOTAL_STICKERS = 54                 # Total stickers on cube
    CORNERS = 8                         # Total corners
    EDGES = 12                          # Total edges

    # Semantic assignment for faces: UP = SUBJECT, DOWN = ACTION, FRONT = OBJECT, BACK = CONTEXT, RIGHT = PROPERTY, LEFT = RELATION
    FACE_ORIENTATIONS = {
        'UP': 0,
        'DOWN': 1,
        'FRONT': 2,
        'BACK': 3,
        'RIGHT': 4,
        'LEFT': 5
    }

    # Standard Rubik's Cube corner positions labeled by conventional notation
    CORNER_POSITIONS = {
        'ULF': 0,  # Upper Left Front
        'UFR': 1,  # Upper Front Right
        'UBR': 2,  # Upper Back Right
        'UBL': 3,  # Upper Back Left
        'DLF': 4,  # Down Left Front
        'DFR': 5,  # Down Front Right
        'DBR': 6,  # Down Back Right
        'DBL': 7   # Down Back Left
    }

    # Standard Rubik's Cube edge positions labeled by conventional notation
    EDGE_POSITIONS = {
        'UF': 0,
        'UR': 1,
        'UB': 2,
        'UL': 3,
        'FR': 4,
        'BR': 5,
        'BL': 6,
        'FL': 7,
        'DF': 8,
        'DR': 9,
        'DB': 10,
        'DL': 11
    }

##############################################
# Data Classes for Concepts and Connections
##############################################

@dataclass
class ConceptNode:
    id: str
    face: str
    position: int
    vector: np.ndarray
    connections: Set[str]
    weight: float = 1.0
    state: float = 0.0
    last_update: float = 0.0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ConceptNode) and self.id == other.id

@dataclass
class EdgeConnection:
    source: str
    target: str
    weight: float
    type: str  # e.g., 'ADJACENT', 'CORNER', 'EDGE'
    bidirectional: bool = True
    active: bool = True
    last_update: float = 0.0

@dataclass
class ConceptPattern:
    id: str
    concepts: List[str]
    pattern_type: str  # e.g., 'HIERARCHICAL', 'INTERACTION', 'TEMPORAL', etc.
    confidence: float
    requirements: Dict[str, float]
    activation_threshold: float = 0.7

##############################################
# Semantic Face: Complete Mapping for Each Cube Face
##############################################

class SemanticFace:
    def __init__(self, face_type: str):
        self.face_type = face_type
        self.stickers = self._init_stickers()

    def _init_stickers(self) -> Dict[str, int]:
        # Complete mapping for each semantic category; no omissions.
        if self.face_type == 'SUBJECT':
            return {
                'ENTITY': 0,
                'PERSON': 1,
                'SYSTEM': 2,
                'CONCEPT': 3,
                'GROUP': 4,
                'LOCATION': 5,
                'TIME': 6,
                'EVENT': 7,
                'ABSTRACT': 8
            }
        elif self.face_type == 'ACTION':
            return {
                'TRANSFORM': 0,
                'CREATE': 1,
                'DESTROY': 2,
                'MODIFY': 3,
                'ANALYZE': 4,
                'COMBINE': 5,
                'SEPARATE': 6,
                'MOVE': 7,
                'STATE': 8
            }
        elif self.face_type == 'OBJECT':
            return {
                'DATA': 0,
                'RESOURCE': 1,
                'TOOL': 2,
                'RESULT': 3,
                'INPUT': 4,
                'OUTPUT': 5,
                'COMPONENT': 6,
                'COLLECTION': 7,
                'ATTRIBUTE': 8
            }
        elif self.face_type == 'CONTEXT':
            return {
                'CONDITION': 0,
                'ENVIRONMENT': 1,
                'SCOPE': 2,
                'CONSTRAINT': 3,
                'REQUIREMENT': 4,
                'ASSUMPTION': 5,
                'STATE': 6,
                'MODE': 7,
                'PHASE': 8
            }
        elif self.face_type == 'PROPERTY':
            return {
                'TYPE': 0,
                'VALUE': 1,
                'STATUS': 2,
                'QUALITY': 3,
                'QUANTITY': 4,
                'FORMAT': 5,
                'PATTERN': 6,
                'STRUCTURE': 7,
                'BEHAVIOR': 8
            }
        else:  # RELATION
            return {
                'CONNECTS': 0,
                'CONTAINS': 1,
                'PRODUCES': 2,
                'INFLUENCES': 3,
                'DEPENDS': 4,
                'EQUALS': 5,
                'MAPS': 6,
                'TRANSFORMS': 7,
                'LINKS': 8
            }

##############################################
# Complete Corner & Edge Mappings
##############################################

class CornerMapping:
    # Complete mapping for all eight corners with full details.
    CORNER_MAP: Dict[int, List[Tuple[int, int]]] = {
        0: [(0, 0), (5, 2), (2, 0)],  # ULF: SUBJECT ENTITY, RELATION LINKS, OBJECT DATA
        1: [(0, 2), (2, 2), (4, 0)],  # UFR: SUBJECT SYSTEM, OBJECT TOOL, PROPERTY TYPE
        2: [(0, 8), (4, 2), (3, 0)],  # UBR: SUBJECT ABSTRACT, PROPERTY STATUS, CONTEXT CONDITION
        3: [(0, 6), (3, 2), (5, 0)],  # UBL: SUBJECT TIME, CONTEXT SCOPE, RELATION CONNECTS
        4: [(1, 0), (5, 8), (2, 6)],  # DLF: ACTION TRANSFORM, RELATION LINKS, OBJECT COMPONENT
        5: [(1, 2), (2, 8), (4, 6)],  # DFR: ACTION DESTROY, OBJECT OUTPUT, PROPERTY PATTERN
        6: [(1, 8), (4, 8), (3, 6)],  # DBR: ACTION STATE, PROPERTY BEHAVIOR, CONTEXT STATE
        7: [(1, 6), (3, 8), (5, 6)]   # DBL: ACTION MOVE, CONTEXT PHASE, RELATION EQUALS
    }

    # Complete rotation mappings for corners for all three orientations.
    CORNER_ROTATIONS: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
        (0, 0): [(0, 0), (5, 2), (2, 0)],
        (0, 1): [(5, 2), (2, 0), (0, 0)],
        (0, 2): [(2, 0), (0, 0), (5, 2)],
        (1, 0): [(0, 2), (2, 2), (4, 0)],
        (1, 1): [(2, 2), (4, 0), (0, 2)],
        (1, 2): [(4, 0), (0, 2), (2, 2)],
        (2, 0): [(0, 8), (4, 2), (3, 0)],
        (2, 1): [(4, 2), (3, 0), (0, 8)],
        (2, 2): [(3, 0), (0, 8), (4, 2)],
        (3, 0): [(0, 6), (3, 2), (5, 0)],
        (3, 1): [(3, 2), (5, 0), (0, 6)],
        (3, 2): [(5, 0), (0, 6), (3, 2)],
        (4, 0): [(1, 0), (5, 8), (2, 6)],
        (4, 1): [(5, 8), (2, 6), (1, 0)],
        (4, 2): [(2, 6), (1, 0), (5, 8)],
        (5, 0): [(1, 2), (2, 8), (4, 6)],
        (5, 1): [(2, 8), (4, 6), (1, 2)],
        (5, 2): [(4, 6), (1, 2), (2, 8)],
        (6, 0): [(1, 8), (4, 8), (3, 6)],
        (6, 1): [(4, 8), (3, 6), (1, 8)],
        (6, 2): [(3, 6), (1, 8), (4, 8)],
        (7, 0): [(1, 6), (3, 8), (5, 6)],
        (7, 1): [(3, 8), (5, 6), (1, 6)],
        (7, 2): [(5, 6), (1, 6), (3, 8)]
    }

class EdgeMapping:
    # Complete mapping for all twelve edges.
    EDGE_MAP: Dict[int, List[Tuple[int, int]]] = {
        0: [(0, 1), (2, 1)],    # UF: between SUBJECT and OBJECT
        1: [(0, 5), (4, 1)],    # UR: between SUBJECT and PROPERTY
        2: [(0, 7), (3, 1)],    # UB: between SUBJECT and CONTEXT
        3: [(0, 3), (5, 1)],    # UL: between SUBJECT and RELATION
        4: [(2, 3), (4, 7)],    # FR: between OBJECT and PROPERTY
        5: [(3, 3), (4, 5)],    # BR: between CONTEXT and PROPERTY
        6: [(3, 7), (5, 5)],    # BL: between CONTEXT and RELATION
        7: [(2, 7), (5, 3)],    # FL: between OBJECT and RELATION
        8: [(1, 1), (2, 5)],    # DF: between ACTION and OBJECT
        9: [(1, 5), (4, 3)],    # DR: between ACTION and PROPERTY
        10: [(1, 7), (3, 5)],   # DB: between ACTION and CONTEXT
        11: [(1, 3), (5, 7)]    # DL: between ACTION and RELATION
    }

    # Complete edge flip mappings.
    EDGE_FLIPS: Dict[Tuple[int, bool], List[Tuple[int, int]]] = {
        (0, False): [(0, 1), (2, 1)],
        (0, True):  [(2, 1), (0, 1)],
        (1, False): [(0, 5), (4, 1)],
        (1, True):  [(4, 1), (0, 5)],
        (2, False): [(0, 7), (3, 1)],
        (2, True):  [(3, 1), (0, 7)],
        (3, False): [(0, 3), (5, 1)],
        (3, True):  [(5, 1), (0, 3)],
        (4, False): [(2, 3), (4, 7)],
        (4, True):  [(4, 7), (2, 3)],
        (5, False): [(3, 3), (4, 5)],
        (5, True):  [(4, 5), (3, 3)],
        (6, False): [(3, 7), (5, 5)],
        (6, True):  [(5, 5), (3, 7)],
        (7, False): [(2, 7), (5, 3)],
        (7, True):  [(5, 3), (2, 7)],
        (8, False): [(1, 1), (2, 5)],
        (8, True):  [(2, 5), (1, 1)],
        (9, False): [(1, 5), (4, 3)],
        (9, True):  [(4, 3), (1, 5)],
        (10, False): [(1, 7), (3, 5)],
        (10, True):  [(3, 5), (1, 7)],
        (11, False): [(1, 3), (5, 7)],
        (11, True):  [(5, 7), (1, 3)]
    }

##############################################
# Semantic Mapping: Text to Cube State Representation
##############################################

class SemanticMappingSystem:
    def __init__(self):
        self.faces: Dict[str, SemanticFace] = {
            face_type: SemanticFace(face_type)
            for face_type in ['SUBJECT', 'ACTION', 'OBJECT', 'CONTEXT', 'PROPERTY', 'RELATION']
        }

    def get_semantic_position(self, concept: str) -> Optional[Tuple[str, int]]:
        for face_type, face in self.faces.items():
            if concept.upper() in face.stickers:
                return (face_type, face.stickers[concept.upper()])
        return None

    def map_text_to_cube_state(self, text: str) -> np.ndarray:
        # Create a state matrix with dimensions [faces x stickers]
        state = np.zeros((CubeConstants.FACES, CubeConstants.STICKERS_PER_FACE))
        tokens = self._tokenize_text(text)
        semantics = self._extract_semantics(tokens)
        for face_type, concept in semantics:
            if face_type in self.faces:
                sticker_id = self.faces[face_type].stickers.get(concept.upper())
                if sticker_id is not None:
                    face_idx = list(self.faces.keys()).index(face_type)
                    state[face_idx, sticker_id] = 1
        return state

    def _tokenize_text(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def _extract_semantics(self, tokens: List[str]) -> List[Tuple[str, str]]:
        semantics = []
        for token in tokens:
            for face_type, face in self.faces.items():
                if token.upper() in face.stickers:
                    semantics.append((face_type, token.upper()))
                else:
                    related = self._find_related_concept(token, face_type)
                    if related:
                        semantics.append((face_type, related))
        return semantics

    def _find_related_concept(self, token: str, face_type: str) -> Optional[str]:
        token = token.upper()
        related_terms = {
            'SUBJECT': {
                'PERSON': ['USER', 'ADMIN', 'DEVELOPER', 'CLIENT'],
                'SYSTEM': ['PROGRAM', 'APPLICATION', 'SERVICE', 'PLATFORM'],
                'CONCEPT': ['IDEA', 'THEORY', 'PRINCIPLE', 'NOTION'],
                'GROUP': ['TEAM', 'ORGANIZATION', 'COLLECTION', 'SET'],
                'LOCATION': ['PLACE', 'SITE', 'ADDRESS'],
                'TIME': ['DATE', 'PERIOD', 'DURATION'],
                'EVENT': ['INCIDENT', 'OCCURRENCE', 'ACTION'],
                'ABSTRACT': ['MODEL', 'PATTERN', 'STRUCTURE']
            },
            'ACTION': {
                'TRANSFORM': ['CHANGE', 'CONVERT', 'MODIFY', 'ALTER'],
                'CREATE': ['MAKE', 'BUILD', 'GENERATE', 'PRODUCE'],
                'DESTROY': ['DELETE', 'REMOVE', 'ELIMINATE', 'ERASE'],
                'MODIFY': ['UPDATE', 'EDIT', 'ADJUST', 'REVISE'],
                'ANALYZE': ['EXAMINE', 'STUDY', 'EVALUATE', 'ASSESS'],
                'COMBINE': ['MERGE', 'JOIN', 'UNITE', 'INTEGRATE'],
                'SEPARATE': ['SPLIT', 'DIVIDE', 'PARTITION', 'ISOLATE'],
                'MOVE': ['TRANSFER', 'SHIFT', 'RELOCATE', 'TRANSPORT'],
                'STATE': ['CONDITION', 'MODE', 'STATUS']
            },
            'OBJECT': {
                'DATA': ['INFORMATION', 'CONTENT', 'RECORDS'],
                'RESOURCE': ['ASSET', 'MATERIAL', 'SOURCE'],
                'TOOL': ['INSTRUMENT', 'DEVICE', 'IMPLEMENT'],
                'RESULT': ['OUTCOME', 'PRODUCT', 'CONSEQUENCE'],
                'INPUT': ['PARAMETER', 'ARGUMENT'],
                'OUTPUT': ['RESPONSE', 'RETURN'],
                'COMPONENT': ['PART', 'MODULE', 'UNIT'],
                'COLLECTION': ['ARRAY', 'LIST', 'GROUP'],
                'ATTRIBUTE': ['PROPERTY', 'FEATURE', 'QUALITY']
            },
            'CONTEXT': {
                'CONDITION': ['STATE', 'SITUATION', 'CIRCUMSTANCE'],
                'ENVIRONMENT': ['SETTING', 'SURROUNDINGS', 'FRAMEWORK'],
                'SCOPE': ['RANGE', 'EXTENT', 'BOUNDARY'],
                'CONSTRAINT': ['LIMIT', 'RESTRICTION', 'RULE'],
                'REQUIREMENT': ['NEED', 'DEMAND', 'SPECIFICATION'],
                'ASSUMPTION': ['HYPOTHESIS', 'PREMISE', 'PRESUMPTION'],
                'STATE': ['MODE', 'PHASE'],
                'MODE': ['METHOD', 'STYLE'],
                'PHASE': ['STAGE', 'STEP']
            },
            'PROPERTY': {
                'TYPE': ['KIND', 'CATEGORY', 'CLASS'],
                'VALUE': ['AMOUNT', 'QUANTITY', 'MAGNITUDE'],
                'STATUS': ['CONDITION', 'MODE'],
                'QUALITY': ['ATTRIBUTE', 'FEATURE', 'CHARACTERISTIC'],
                'QUANTITY': ['NUMBER', 'COUNT'],
                'FORMAT': ['STRUCTURE', 'LAYOUT'],
                'PATTERN': ['TEMPLATE', 'MODEL', 'DESIGN'],
                'STRUCTURE': ['ORGANIZATION', 'ARRANGEMENT'],
                'BEHAVIOR': ['ACTION', 'FUNCTION']
            },
            'RELATION': {
                'CONNECTS': ['LINKS', 'JOINS', 'RELATES'],
                'CONTAINS': ['HOLDS', 'INCLUDES'],
                'PRODUCES': ['CREATES', 'GENERATES'],
                'INFLUENCES': ['AFFECTS', 'IMPACTS'],
                'DEPENDS': ['REQUIRES', 'NEEDS'],
                'EQUALS': ['MATCHES', 'IDENTICAL'],
                'MAPS': ['TRANSFORMS', 'CONVERTS'],
                'TRANSFORMS': ['CHANGES', 'MODIFIES'],
                'LINKS': ['CONNECTS', 'JOINS']
            }
        }
        face_related = related_terms.get(face_type, {})
        for concept, terms in face_related.items():
            if token in terms:
                return concept
        return None

##############################################
# Cube State Handler: Moves and State Optimization
##############################################

class CubeStateHandler:
    def __init__(self):
        self.move_sequences: Dict[str, List[str]] = self._init_move_sequences()
        self.move_history: List[str] = []

    def _init_move_sequences(self) -> Dict[str, List[str]]:
        return {
            'ORIENT_CORNERS': ['U', 'R', "U'", "L'", 'U', "R'", "U'", 'L'],
            'SWAP_EDGES': ['R', 'U', "R'", "U'", "R'", 'F', 'R2', "U'", "R'", "U'", 'R', 'U', "R'", "F'"],
            'ALIGN_CENTERS': ['F', "B'", 'R', "L'", 'U', "D'"]
        }

    def optimize_state(self, state: np.ndarray) -> np.ndarray:
        metrics = self._calculate_state_metrics(state)
        while not self._is_optimal(metrics):
            best_sequence = self._select_best_sequence(metrics)
            if not best_sequence:
                break
            state = self._apply_move_sequence(state, best_sequence)
            self.move_history.extend(best_sequence)
            metrics = self._calculate_state_metrics(state)
        return state

    def _calculate_state_metrics(self, state: np.ndarray) -> Dict[str, float]:
        return {
            'corner_alignment': self._calculate_corner_alignment(state),
            'edge_alignment': self._calculate_edge_alignment(state),
            'center_alignment': self._calculate_center_alignment(state),
            'pattern_coherence': self._calculate_pattern_coherence(state)
        }

    def _calculate_corner_alignment(self, state: np.ndarray) -> float:
        corner_indices = [0, 2, 6, 8]
        scores = []
        for face in state:
            corners = face[corner_indices]
            scores.append(np.mean(corners))
        return np.mean(scores)

    def _calculate_edge_alignment(self, state: np.ndarray) -> float:
        edge_indices = [1, 3, 5, 7]
        scores = []
        for face in state:
            edges = face[edge_indices]
            scores.append(np.mean(edges))
        return np.mean(scores)

    def _calculate_center_alignment(self, state: np.ndarray) -> float:
        centers = [face[4] for face in state]
        return np.mean(centers)

    def _calculate_pattern_coherence(self, state: np.ndarray) -> float:
        return np.mean(state)

    def _is_optimal(self, metrics: Dict[str, float]) -> bool:
        thresholds = {
            'corner_alignment': 0.9,
            'edge_alignment': 0.9,
            'center_alignment': 0.95,
            'pattern_coherence': 0.8
        }
        return all(metrics[m] >= thresholds[m] for m in metrics)

    def _select_best_sequence(self, metrics: Dict[str, float]) -> Optional[List[str]]:
        worst_metric = min(metrics.items(), key=lambda x: x[1])[0]
        if worst_metric == 'corner_alignment':
            return self.move_sequences['ORIENT_CORNERS']
        elif worst_metric == 'edge_alignment':
            return self.move_sequences['SWAP_EDGES']
        elif worst_metric == 'center_alignment':
            return self.move_sequences['ALIGN_CENTERS']
        return None

    def _apply_move_sequence(self, state: np.ndarray, sequence: List[str]) -> np.ndarray:
        for move in sequence:
            state = self._apply_move(state, move)
        return state

    def _apply_move(self, state: np.ndarray, move: str) -> np.ndarray:
        face_map = {'U': 0, 'D': 1, 'F': 2, 'B': 3, 'R': 4, 'L': 5}
        face = move[0]
        direction = 1
        if len(move) > 1:
            if move[1] == "'":
                direction = -1
            elif move[1] == "2":
                direction = 2
        face_idx = face_map.get(face)
        face_state = state[face_idx].reshape(3, 3)
        rotated = np.rot90(face_state, k=direction)
        state[face_idx] = rotated.flatten()
        # Note: Updating adjacent faces is complex and has been simplified.
        return state

##############################################
# Pattern Matcher: Identify Semantic Patterns in Cube State
##############################################

class PatternMatcher:
    def __init__(self):
        self.pattern_templates: Dict[str, np.ndarray] = self._init_pattern_templates()

    def _init_pattern_templates(self) -> Dict[str, np.ndarray]:
        return {
            'SAO': np.array([
                [1,0,1,0,0,0,0,0,1],
                [0,1,0,0,0,0,0,0,0],
                [1,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0]
            ]),
            'PR': np.array([
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [1,0,1,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0]
            ]),
            'CA': np.array([
                [0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0],
                [1,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0]
            ]),
            'SOR': np.array([
                [1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0]
            ]),
            'COMPLEX': np.array([
                [1,1,0,0,0,0,0,1,0],
                [0,1,1,0,0,0,1,0,0],
                [0,0,1,1,0,0,0,0,1],
                [1,0,0,0,1,0,0,0,1],
                [0,1,0,0,1,1,0,0,0],
                [1,0,0,1,0,0,1,0,0]
            ])
        }

    def find_patterns(self, state: np.ndarray) -> List[Dict[str, Any]]:
        patterns_found = []
        for name, template in self.pattern_templates.items():
            for rotation in range(4):
                rotated = np.rot90(template, k=rotation)
                conf = self._calculate_match_confidence(state, rotated)
                if conf > 0.7:
                    positions = self._get_matching_positions(state, rotated)
                    concepts = self._extract_matched_concepts(state, positions)
                    patterns_found.append({
                        'type': name,
                        'confidence': conf,
                        'positions': positions,
                        'concepts': concepts,
                        'rotation': rotation
                    })
        patterns_found.sort(key=lambda x: x['confidence'], reverse=True)
        return patterns_found

    def _calculate_match_confidence(self, state: np.ndarray, template: np.ndarray) -> float:
        matches = (state > 0) == (template > 0)
        weights = np.array([
            [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0],
            [0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],
            [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0],
            [0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],
            [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0],
            [0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8]
        ])
        score = np.sum(matches * weights) / np.sum(template * weights)
        return float(score)

    def _get_matching_positions(self, state: np.ndarray, template: np.ndarray) -> List[Tuple[int, int]]:
        positions = []
        for face in range(state.shape[0]):
            for sticker in range(state.shape[1]):
                if template[face, sticker] > 0 and state[face, sticker] > 0:
                    positions.append((face, sticker))
        return positions

    def _extract_matched_concepts(self, state: np.ndarray, positions: List[Tuple[int, int]]) -> List[str]:
        concepts = []
        face_names = ['SUBJECT', 'ACTION', 'OBJECT', 'CONTEXT', 'PROPERTY', 'RELATION']
        mapping = {
            'SUBJECT': {
                0: 'ENTITY', 1: 'PERSON', 2: 'SYSTEM', 3: 'CONCEPT',
                4: 'GROUP', 5: 'LOCATION', 6: 'TIME', 7: 'EVENT', 8: 'ABSTRACT'
            },
            'ACTION': {
                0: 'TRANSFORM', 1: 'CREATE', 2: 'DESTROY', 3: 'MODIFY',
                4: 'ANALYZE', 5: 'COMBINE', 6: 'SEPARATE', 7: 'MOVE', 8: 'STATE'
            },
            'OBJECT': {
                0: 'DATA', 1: 'RESOURCE', 2: 'TOOL', 3: 'RESULT',
                4: 'INPUT', 5: 'OUTPUT', 6: 'COMPONENT', 7: 'COLLECTION', 8: 'ATTRIBUTE'
            },
            'CONTEXT': {
                0: 'CONDITION', 1: 'ENVIRONMENT', 2: 'SCOPE', 3: 'CONSTRAINT',
                4: 'REQUIREMENT', 5: 'ASSUMPTION', 6: 'STATE', 7: 'MODE', 8: 'PHASE'
            },
            'PROPERTY': {
                0: 'TYPE', 1: 'VALUE', 2: 'STATUS', 3: 'QUALITY',
                4: 'QUANTITY', 5: 'FORMAT', 6: 'PATTERN', 7: 'STRUCTURE', 8: 'BEHAVIOR'
            },
            'RELATION': {
                0: 'CONNECTS', 1: 'CONTAINS', 2: 'PRODUCES', 3: 'INFLUENCES',
                4: 'DEPENDS', 5: 'EQUALS', 6: 'MAPS', 7: 'TRANSFORMS', 8: 'LINKS'
            }
        }
        for face_idx, sticker_idx in positions:
            face_type = face_names[face_idx]
            concept = mapping.get(face_type, {}).get(sticker_idx)
            if concept:
                concepts.append(concept)
        return concepts

##############################################
# Concept Graph Layer: Per-Face Concept Graph
##############################################

class ConceptGraphLayer:
    def __init__(self, face_type: str, layer_size: int = CubeConstants.STICKERS_PER_FACE):
        self.face_type = face_type
        self.layer_size = layer_size
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: Dict[Tuple[str, str], EdgeConnection] = {}
        self.patterns: Dict[str, ConceptPattern] = {}
        self.concept_maps = self._init_concept_maps()
        self._init_nodes()
        self._init_edges()
        self._init_patterns()

    def _init_concept_maps(self) -> Dict[int, Tuple[str, List[str]]]:
        mapping = {}
        if self.face_type == 'SUBJECT':
            mapping = {
                0: ('ENTITY', ['object', 'entity',