import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import GPT2Config, GPT2LMHeadModel

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class GPT2SequenceParallel(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        
        # Load full GPT2 model config
        self.full_model = GPT2LMHeadModel(config)
        
        # Sequence parallelism: split the transformer layers among GPUs along the layer dimension
        # Here we split the transformer blocks (config.n_layer) across world_size GPUs
        total_layers = config.n_layer
        per_rank_layers = total_layers // world_size
        
        start_layer = per_rank_layers * rank
        end_layer = start_layer + per_rank_layers
        
        # Extract only the layers for this rank
        self.layers = nn.ModuleList(self.full_model.transformer.h[start_layer:end_layer])
        self.wte = self.full_model.transformer.wte
        self.wpe = self.full_model.transformer.wpe
        self.ln_f = self.full_model.transformer.ln_f
        self.lm_head = self.full_model.lm_head
        
    def forward(self, input_ids):
        # Embed tokens and positions
        input_embeds = self.wte(input_ids) + self.wpe(torch.arange(input_ids.size(1), device=input_ids.device))
        
        # Send input_embeds to first rank only
        # Each rank applies its own slice of the transformer sequentially
        # Then, communicate outputs across ranks
        
        # Step 1: Broadcast input_embeds from rank 0 to all
        dist.broadcast(input_embeds, src=0)
        
        # Each rank processes its own layers sequentially
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]  # layer output is (hidden_states, attn_weights)
        
        # Step 2: Reduce sum of all partial outputs to rank 0 for final prediction
        # Since layers are partitioned, we gather the output sequentially via send/recv or reduce
        # For simplicity, we just gather data back to rank 0 by all-reduce sum (works if they add to the same tensor)
        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM)
        
        # Only rank 0 does lm_head and returns output
        if self.rank == 0:
            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits
        else:
            return None

def train(rank, world_size):
    print(f"Starting training on rank {rank}.")
    setup(rank, world_size)
    
    # Load GPT-2 config and create sequence parallel model
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2SequenceParallel(config, world_size, rank).cuda(rank)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Dummy input batch: batch_size=2, seq_len=16
    batch_size, seq_len = 2, 16
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda(rank)
    targets = inputs.clone()
    
    model.train()
    optimizer.zero_grad()
    
    logits = model(inputs)
    
    if rank == 0:
        loss_fn = nn.CrossEntropyLoss()
        # reshape logits and targets for loss
        logits = logits.view(-1, config.vocab_size)
        targets = targets.view(-1)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        print(f"Rank {rank} loss: {loss.item()}")
    
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
