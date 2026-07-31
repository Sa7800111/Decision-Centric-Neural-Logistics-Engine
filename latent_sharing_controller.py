class LatentSharingController:
    def __init__(self):
        self.latent_map = {}
        self.latent_values = {}

    def link_latents(self, node_a, node_b, latent_id):
        if not isinstance(node_a, str) or not isinstance(node_b, str):
            raise TypeError()
        self.latent_map[node_a] = latent_id
        self.latent_map[node_b] = latent_id
        if latent_id not in self.latent_values:
            self.latent_values[latent_id] = None

    def get_shared_nodes(self, latent_id):
        if latent_id not in self.latent_values:
            raise KeyError()
        return [k for k, v in self.latent_map.items() if v == latent_id]

    def set_latent_value(self, latent_id, value):
        if latent_id not in self.latent_values:
            raise KeyError()
        self.latent_values[latent_id] = value

    def resolve_latents(self, nodes):
        if not isinstance(nodes, list):
            raise TypeError()
        return {n: self.latent_values.get(self.latent_map.get(n)) for n in nodes}