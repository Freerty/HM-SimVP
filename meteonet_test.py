"""MeteoNet pooled and lead-time evaluation; no plotting or baselines."""
import argparse
from pathlib import Path
import torch
from meteonet_train import (RUN_CONFIG, build_model, build_loss, MeteoNetH5Dataset,
    make_loader, evaluate, format_metrics, write_json)

def main():
    parser = argparse.ArgumentParser(description='Evaluate HM-SimVP on the MeteoNet test/images H5 group.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='meteonet_metrics.json')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = dict(RUN_CONFIG); config.update(state.get('config', {}))
    config.update(data_file=args.data, num_workers=args.workers, use_amp=device.type == 'cuda',
                  metric_thresholds=[.1,1.,5.,10.,20.,30.])
    dataset = MeteoNetH5Dataset(args.data, 'test', config['input_length'], config['output_length'],
                               config['raw_clip_max'], augment=False)
    loader = make_loader(dataset, args.batch_size, False, config)
    model = build_model(config).to(device)
    model.load_state_dict(state.get('model_state_dict', state), strict=True)
    result = evaluate(model, loader, build_loss(config).to(device), device, config, 'MeteoNet test')
    result['checkpoint'] = str(Path(args.checkpoint).resolve())
    result['samples'] = len(dataset)
    write_json(Path(args.output), result)
    print(format_metrics('Test', result))

if __name__ == '__main__':
    main()
