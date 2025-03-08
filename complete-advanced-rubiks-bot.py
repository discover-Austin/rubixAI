import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import json
import re
from collections import defaultdict

class AdvancedRubiksNLUBot:
    """
    An advanced conversational AI inspired by the Rubik's Cube principle:
    Even highly complex conversational states can be navigated efficiently
    with the right set of transformations, similar to how any Rubik's Cube
    configuration can be solved in 20 moves or fewer.
    
    This enhanced version includes:
    - Structured knowledge representation
    - Multi-hop reasoning
    - Context-aware responses
    - Knowledge graph navigation
    - Dynamic response synthesis
    """
    
    def __init__(self, knowledge_base=None, knowledge_file=None, max_moves=10):
        """
        Initialize the bot with a knowledge base and maximum number of reasoning moves.
        
        Args:
            knowledge_base: List of strings or dict of structured knowledge
            knowledge_file: Path to JSON file containing structured knowledge
            max_moves: Maximum number of reasoning transformations
        """
        if knowledge_file:
            with open(knowledge_file, 'r') as f:
                self.structured_knowledge = json.load(f)
        elif isinstance(knowledge_base, dict):
            self.structured_knowledge = knowledge_base
        else:
            self.structured_knowledge = self._convert_to_structured(knowledge_base or [])
            
        # Extract flat knowledge for vectorization
        self.knowledge_base = self._flatten_knowledge()
        
        self.max_moves = max_moves
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        if self.knowledge_base:
            self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_base)
        else:
            # Initialize with empty corpus if no knowledge is provided
            self.vectorizer.fit(["Empty knowledge base"])
            self.knowledge_vectors = None
            
        self.conversation_history = []
        self.knowledge_graph = self._build_knowledge_graph()
        
    def _convert_to_structured(self, knowledge_list):
        """Convert flat knowledge list to structured format"""
        structured = {
            "concepts": {},
            "relationships": []
        }
        
        # Create concepts from knowledge items
        for i, knowledge in enumerate(knowledge_list):
            concept_id = f"c{i}"
            structured["concepts"][concept_id] = {
                "name": self._extract_main_concept(knowledge),
                "description": knowledge,
                "attributes": {},
                "category": self._categorize_knowledge(knowledge)
            }
        
        # Create relationships between concepts
        if knowledge_list:
            knowledge_vectors = TfidfVectorizer().fit_transform(knowledge_list)
            similarities = cosine_similarity(knowledge_vectors)
            
            for i in range(len(knowledge_list)):
                for j in range(i+1, len(knowledge_list)):
                    if similarities[i, j] > 0.2:
                        structured["relationships"].append({
                            "source": f"c{i}",
                            "target": f"c{j}",
                            "type": "related",
                            "strength": float(similarities[i, j])
                        })
        
        return structured
    
    def _extract_main_concept(self, text):
        """Extract the main concept from a piece of text"""
        # A simple implementation - in a real system, use NLP techniques
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        if words:
            return words[0]
        words = text.split()
        if words:
            # Return the first non-stopword
            stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'by', 'to', 'for', 'with'}
            for word in words:
                if word.lower() not in stopwords:
                    return word
        return text[:10] + "..."
    
    def _categorize_knowledge(self, text):
        """Categorize a piece of knowledge"""
        categories = {
            "rubiks_cube": ["rubik", "cube", "puzzle", "solving", "combination"],
            "ai": ["ai", "artificial intelligence", "machine learning", "model", "algorithm"],
            "nlp": ["nlp", "natural language", "text", "processing", "semantic"],
            "graph_theory": ["graph", "network", "node", "edge", "path"]
        }
        
        text_lower = text.lower()
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        
        return "general"
    
    def _flatten_knowledge(self):
        """Extract flat text from structured knowledge for vectorization"""
        flattened = []
        
        for concept_id, concept in self.structured_knowledge["concepts"].items():
            if "description" in concept:
                flattened.append(concept["description"])
                
            # Also include attributes as separate knowledge pieces
            for attr_name, attr_value in concept.get("attributes", {}).items():
                if isinstance(attr_value, str):
                    flattened.append(f"{concept['name']} {attr_name}: {attr_value}")
        
        return flattened
    
    def _build_knowledge_graph(self):
        """Build a networkx graph from the structured knowledge"""
        G = nx.Graph()
        
        # Add concept nodes
        for concept_id, concept in self.structured_knowledge["concepts"].items():
            G.add_node(concept_id, 
                      name=concept.get("name", ""),
                      description=concept.get("description", ""),
                      category=concept.get("category", "general"))
        
        # Add relationship edges
        for rel in self.structured_knowledge["relationships"]:
            G.add_edge(rel["source"], rel["target"], 
                      type=rel.get("type", "related"),
                      strength=rel.get("strength", 0.5))
        
        return G
    
    def query_to_state(self, query):
        """Transform a user query into a state vector"""
        query_vector = self.vectorizer.transform([query])
        return query_vector
    
    def find_relevant_knowledge(self, query_vector, top_k=5):
        """Find the most relevant knowledge pieces"""
        if self.knowledge_vectors is None or self.knowledge_vectors.shape[0] == 0:
            return []
            
        similarities = cosine_similarity(query_vector, self.knowledge_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(self.knowledge_base[i], similarities[i], i) for i in top_indices if similarities[i] > 0.1]
    
    def build_reasoning_graph(self, query_vector, target_concepts=None):
        """
        Build a graph of possible reasoning paths from the current query 
        state to an answer that incorporates target concepts
        """
        G = nx.DiGraph()
        
        # Add the starting node (query state)
        G.add_node("query", vector=query_vector, type="query")
        
        # Add knowledge nodes
        for i, knowledge in enumerate(self.knowledge_base):
            knowledge_vector = self.knowledge_vectors[i]
            concept_id = self._find_concept_id_for_knowledge(knowledge)
            category = "unknown"
            
            if concept_id:
                category = self.structured_knowledge["concepts"][concept_id].get("category", "unknown")
                
            G.add_node(f"k{i}", 
                      vector=knowledge_vector, 
                      text=knowledge, 
                      type="knowledge",
                      concept_id=concept_id,
                      category=category)
            
            # Connect query to knowledge based on relevance
            sim = cosine_similarity(query_vector, knowledge_vector)[0][0]
            if sim > 0.1:  # Lower threshold to include more potential paths
                G.add_edge("query", f"k{i}", weight=sim, type="relevance")
        
        # Connect knowledge nodes to each other based on relationships in the structured knowledge
        for rel in self.structured_knowledge["relationships"]:
            source_id = rel["source"]
            target_id = rel["target"]
            strength = rel.get("strength", 0.5)
            rel_type = rel.get("type", "related")
            
            # Find knowledge nodes corresponding to these concepts
            source_knowledge = [f"k{i}" for i, k in enumerate(self.knowledge_base) 
                               if self._find_concept_id_for_knowledge(k) == source_id]
            target_knowledge = [f"k{i}" for i, k in enumerate(self.knowledge_base) 
                               if self._find_concept_id_for_knowledge(k) == target_id]
            
            # Connect them
            for s in source_knowledge:
                for t in target_knowledge:
                    if s != t:
                        G.add_edge(s, t, weight=strength, type=rel_type)
                        G.add_edge(t, s, weight=strength, type=rel_type)  # Add bidirectional edge
        
        # Add context-based edges from conversation history
        if self.conversation_history:
            # Get vector for the most recent conversation history
            history_vector = self.vectorizer.transform([self.conversation_history[-1]])
            
            # Add history node
            G.add_node("history", vector=history_vector, type="history")
            
            # Connect history to knowledge nodes
            for i, knowledge in enumerate(self.knowledge_base):
                knowledge_vector = self.knowledge_vectors[i]
                sim = cosine_similarity(history_vector, knowledge_vector)[0][0]
                if sim > 0.15:  # Slightly higher threshold for history relevance
                    G.add_edge("history", f"k{i}", weight=sim, type="history_relevance")
        
        return G
    
    def _find_concept_id_for_knowledge(self, knowledge):
        """Find the concept ID that corresponds to a piece of knowledge"""
        for concept_id, concept in self.structured_knowledge["concepts"].items():
            if concept.get("description") == knowledge:
                return concept_id
        return None
    
    def find_optimal_reasoning_path(self, query, target_concepts=None):
        """
        Find the optimal sequence of reasoning steps (like Rubik's moves)
        to transform the query into an answer
        """
        query_vector = self.query_to_state(query)
        
        if not target_concepts:
            # If no specific target concepts, find the most relevant knowledge
            relevant_knowledge = self.find_relevant_knowledge(query_vector)
            target_concepts = [k for k, _, _ in relevant_knowledge]
        
        # Build reasoning graph
        G = self.build_reasoning_graph(query_vector, target_concepts)
        
        # Identify potential target nodes (most relevant knowledge)
        relevant_nodes = []
        for node in G.nodes():
            if node.startswith('k') and G.has_edge("query", node):
                relevance = G.edges["query", node]['weight']
                relevant_nodes.append((node, relevance))
        
        # Sort by relevance
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Get top targets
        top_targets = [node for node, _ in relevant_nodes[:5]]
        
        # Find paths to target nodes, limiting to max_moves
        all_paths = []
        for target in top_targets:
            try:
                # Try to find path from query
                path = nx.shortest_path(G, "query", target, weight=lambda u, v, d: 1/d['weight'])
                if len(path) <= self.max_moves + 1:  # +1 because path includes start and end
                    all_paths.append((path, G.edges["query", path[1]]['weight']))
                    
                # If we have conversation history, also try path from history through query
                if "history" in G.nodes():
                    try:
                        history_path = nx.shortest_path(G, "history", target, weight=lambda u, v, d: 1/d['weight'])
                        if len(history_path) <= self.max_moves + 2:  # +2 for history and query
                            # Insert query after history if not already there
                            if history_path[1] != "query":
                                history_path.insert(1, "query")
                            all_paths.append((history_path, G.edges["history", history_path[2]]['weight']))
                    except nx.NetworkXNoPath:
                        pass
                        
            except nx.NetworkXNoPath:
                continue
        
        # If no direct paths found, try multi-hop reasoning
        if not all_paths:
            # Use PageRank to find important nodes in the graph
            if len(G.nodes()) > 2:  # Need at least query node and one knowledge node
                pagerank = nx.pagerank(G)
                important_nodes = sorted([(n, r) for n, r in pagerank.items() if n.startswith('k')], 
                                       key=lambda x: x[1], reverse=True)[:5]
                
                for node, _ in important_nodes:
                    try:
                        path = nx.shortest_path(G, "query", node, weight=lambda u, v, d: 1/d['weight'])
                        if len(path) <= self.max_moves + 1:
                            all_paths.append((path, 0.5))  # Use a default weight
                    except nx.NetworkXNoPath:
                        continue
        
        # Return the best path based on relevance and length
        if not all_paths:
            return None, None
        
        # Score paths by combination of relevance and brevity
        path_scores = {}
        for path, relevance in all_paths:
            # Add a length penalty - shorter paths are generally better
            length_penalty = 1.0 / len(path)
            path_scores[tuple(path)] = relevance * length_penalty
        
        best_path = max(path_scores, key=path_scores.get)
        return list(best_path), G
    
    def extract_knowledge_from_path(self, path, G):
        """Extract and organize knowledge from a reasoning path"""
        knowledge_items = []
        
        # Skip the first node (query) and possibly history node
        start_idx = 1
        if len(path) > 1 and path[1] == "history":
            start_idx = 2
            
        for i in range(start_idx, len(path)):
            node = path[i]
            if node.startswith('k'):
                knowledge_text = G.nodes[node]['text']
                category = G.nodes[node].get('category', 'unknown')
                
                # Get incoming edge weight as relevance
                prev_node = path[i-1]
                relevance = G.edges[prev_node, node]['weight']
                
                knowledge_items.append({
                    'text': knowledge_text,
                    'category': category,
                    'relevance': relevance,
                    'node_id': node
                })
        
        # Sort by relevance
        knowledge_items.sort(key=lambda x: x['relevance'], reverse=True)
        
        return knowledge_items
    
    def organize_knowledge_by_category(self, knowledge_items):
        """Organize knowledge items by category for better response structure"""
        by_category = defaultdict(list)
        
        for item in knowledge_items:
            by_category[item['category']].append(item)
            
        # Sort items within each category by relevance
        for category in by_category:
            by_category[category].sort(key=lambda x: x['relevance'], reverse=True)
            
        return by_category
    
    def generate_response(self, query):
        """
        Generate a response to the user query by finding the optimal reasoning path
        and synthesizing knowledge along that path (like solving a Rubik's Cube)
        """
        self.conversation_history.append(query)
        
        # Find optimal reasoning path
        reasoning_path, G = self.find_optimal_reasoning_path(query)
        
        if not reasoning_path or len(reasoning_path) < 2:
            return "I don't have enough information to answer that question effectively."
        
        # Extract knowledge from the reasoning path
        knowledge_items = self.extract_knowledge_from_path(reasoning_path, G)
        
        if not knowledge_items:
            return "I couldn't find relevant information to answer your question."
        
        # Organize knowledge by category for better response structure
        categorized_knowledge = self.organize_knowledge_by_category(knowledge_items)
        
        # Generate response based on the knowledge structure
        response = self._synthesize_response(query, categorized_knowledge, knowledge_items)
        
        return response
    
    def _synthesize_response(self, query, categorized_knowledge, all_knowledge):
        """Synthesize a coherent response from the knowledge items"""
        if not all_knowledge:
            return "I don't have enough information to answer that question."
            
        # Analyze the question type
        question_types = {
            "what": "descriptive",
            "how": "procedural",
            "why": "explanatory",
            "when": "temporal",
            "where": "spatial",
            "who": "entity",
            "which": "selection",
            "can": "capability",
            "is": "verification",
            "are": "verification",
            "do": "verification",
            "does": "verification"
        }
        
        # Determine question type
        query_lower = query.lower()
        question_type = "general"
        
        for qt, qtype in question_types.items():
            if query_lower.startswith(qt + " ") or f" {qt} " in query_lower:
                question_type = qtype
                break
        
        # Get the top knowledge item
        top_item = all_knowledge[0]['text']
        
        # Now construct response based on question type and knowledge
        if len(all_knowledge) == 1:
            # Simple response with just one knowledge item
            return top_item
            
        # For more complex responses, structure by category and relevance
        response_parts = []
        
        # Start with the most relevant item
        response_parts.append(top_item)
        
        # Add follow-up information from other categories
        categories_used = {all_knowledge[0]['category']}
        transition_phrases = {
            "descriptive": ["Additionally, ", "Furthermore, ", "It's also worth noting that "],
            "procedural": ["The process also involves ", "Another step is to ", "Additionally, "],
            "explanatory": ["This is because ", "Another reason is that ", "Furthermore, "],
            "general": ["Moreover, ", "Additionally, ", "Furthermore, ", "Also, "]
        }
        
        # Get appropriate transition phrases
        transitions = transition_phrases.get(question_type, transition_phrases["general"])
        transition_idx = 0
        
        # Add information from other categories
        for category, items in categorized_knowledge.items():
            if category in categories_used:
                continue
                
            if items:
                # Add the most relevant item from this category
                transition = transitions[transition_idx % len(transitions)]
                response_parts.append(f"{transition}{items[0]['text']}")
                categories_used.add(category)
                transition_idx += 1
                
                # Only add up to 2 more categories to avoid too long responses
                if len(categories_used) >= 3:
                    break
        
        # For explanation type questions, add a concluding statement
        if question_type == "explanatory" and len(response_parts) > 1:
            response_parts.append("This explanation combines multiple aspects to provide a comprehensive answer.")
        
        # Combine parts into a cohesive response
        response = " ".join(response_parts)
        
        return response
    
    def get_reasoning_path_explanation(self, query):
        """
        Explain the reasoning path used to generate a response
        """
        reasoning_path, G = self.find_optimal_reasoning_path(query)
        
        if not reasoning_path or len(reasoning_path) < 2:
            return "Could not find a reasoning path for this query."
        
        explanation = "Reasoning steps:\n"
        
        for i in range(len(reasoning_path)-1):
            from_node = reasoning_path[i]
            to_node = reasoning_path[i+1]
            
            # Get node information
            if from_node == "query":
                from_text = f"Query: {query}"
            elif from_node == "history":
                from_text = f"History: {self.conversation_history[-2] if len(self.conversation_history) > 1 else 'No previous history'}"
            else:
                from_text = f"Knowledge: {G.nodes[from_node]['text']}"
                
            if to_node.startswith('k'):
                to_text = f"Knowledge: {G.nodes[to_node]['text']}"
            else:
                to_text = f"Node: {to_node}"
            
            # Get edge information
            edge_weight = G.edges[from_node, to_node]['weight']
            edge_type = G.edges[from_node, to_node].get('type', 'default')
            
            explanation += f"Step {i+1}: {from_text} -> {to_text} (relevance: {edge_weight:.2f}, type: {edge_type})\n"
        
        return explanation
    
    def add_knowledge(self, knowledge_text, attributes=None, category=None):
        """Add a new piece of knowledge to the bot's knowledge base"""
        # Create a new concept ID
        concept_count = len(self.structured_knowledge["concepts"])
        concept_id = f"c{concept_count}"
        
        # Determine category if not provided
        if category is None:
            category = self._categorize_knowledge(knowledge_text)
            
        # Create concept entry
        self.structured_knowledge["concepts"][concept_id] = {
            "name": self._extract_main_concept(knowledge_text),
            "description": knowledge_text,
            "attributes": attributes or {},
            "category": category
        }
        
        # Update flat knowledge base
        self.knowledge_base.append(knowledge_text)
        
        # Update vectors
        self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_base)
        
        # Update knowledge graph
        self._update_knowledge_graph_with_new_concept(concept_id)
        
        return concept_id
    
    def _update_knowledge_graph_with_new_concept(self, new_concept_id):
        """Update the knowledge graph with a new concept"""
        # Add to knowledge graph
        concept = self.structured_knowledge["concepts"][new_concept_id]
        self.knowledge_graph.add_node(new_concept_id,
                                     name=concept.get("name", ""),
                                     description=concept.get("description", ""),
                                     category=concept.get("category", "general"))
        
        # Calculate relationships with existing concepts
        new_knowledge = concept["description"]
        new_vector = self.vectorizer.transform([new_knowledge])
        
        # Find relationships with existing knowledge
        for concept_id, other_concept in self.structured_knowledge["concepts"].items():
            if concept_id == new_concept_id:
                continue
                
            other_desc = other_concept.get("description", "")
            if other_desc:
                other_vector = self.vectorizer.transform([other_desc])
                similarity = cosine_similarity(new_vector, other_vector)[0][0]
                
                if similarity > 0.2:
                    # Add to structured knowledge
                    self.structured_knowledge["relationships"].append({
                        "source": new_concept_id,
                        "target": concept_id,
                        "type": "related",
                        "strength": float(similarity)
                    })
                    
                    # Add to knowledge graph
                    self.knowledge_graph.add_edge(new_concept_id, concept_id,
                                               type="related",
                                               strength=similarity)

