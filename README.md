## Evaluate the throughput and memory usage of your compressed model
### Quick Start
```bash
cd experiements
sh test_speedup_0.sh # evaluate the memory usage and throughput
```

### Load Your Own Model
Replace the CustomLlamaDecoderLayer module by your own in `replace_decoder_layers` function