import torch

from openfold.np import residue_constants as rc

def modify_ae_model_main(args):
    x = torch.load(args.model_files[0], map_location="cpu", weights_only=False)
    state_dict = x["state_dict"]
    if args.verbose:
        print(state_dict.keys())
    a = state_dict["encoder.init_repr_factory.linear_out.weight"]
    # feats_seq: ["chain_break_per_res", "x1_aatype", "x1_a37coors_nm", "x1_a37coors_nm_rel", "x1_bb_angles", "x1_sidechain_angles", "chain_idx_seq"]  # Sequence features to include in initial representation
    # FROM:
    #     chain_break_per_res: 1                      0:  1
    #     x1_aatype: 20                               1: 21
    #     x1_a37coors_nm: 37 * 3 + 37 * 1 = 148      21:169
    #     x1_a37coors_nm_rel: 37 * 3 + 37 * 1 = 148 169:317
    #     x1_bb_angles: 3 * 21 = 63                 317:380
    #     x1_sidechain_angles: 4 * 21 + 4 = 88      380:468
    #     => 468
    # TO:
    #     chain_break_per_res: 1
    #     x1_aatype: rc.restype_num = 30
    #     x1_a37coors_nm: rc.atom_type_num * 3 + rc.atom_type_num * 1 = 260
    #     x1_a37coors_nm_rel: rc.atom_type_num * 3 + rc.atom_type_num * 1 = 260
    #     x1_bb_angles: 3 * 21 = 63
    #     x1_sidechain_angles: rc.chi_angles_num * 21 + rc.chi_angles_num = 88
    #     => 702
    assert a.shape[1] == 468, a.shape
    b = torch.cat(
        [
            a[:, :21],
            torch.randn(a.shape[0], rc.restype_num - 20),
            a[:, 21:132],
            torch.randn(a.shape[0], (rc.atom_type_num - 37) * 3),
            a[:, 132:169],
            torch.randn(a.shape[0], (rc.atom_type_num - 37) * 1),
            a[:, 169:280],
            torch.randn(a.shape[0], (rc.atom_type_num - 37) * 3),
            a[:, 280:317],
            torch.randn(a.shape[0], (rc.atom_type_num - 37) * 1),
            a[:, 317:464],
            torch.randn(a.shape[0], (rc.chi_angles_num - 4) * 21),
            a[:, 464:468],
            torch.randn(a.shape[0], (rc.chi_angles_num - 4) * 1),
        ],
        dim=1
    )
    print(f"{a.shape} => {b.shape}")
    state_dict["encoder.init_repr_factory.linear_out.weight"] = b

    a = state_dict["decoder.logit_linear.1.weight"]
    assert a.shape[0] == 20, a.shape
    b = torch.cat(
        [a, torch.randn(rc.restype_num - a.shape[0], a.shape[1])], dim=0
    )
    print(f"{a.shape} => {b.shape}")
    state_dict["decoder.logit_linear.1.weight"] = b

    a = state_dict["decoder.struct_linear.1.weight"]
    assert a.shape[0] == 111, a.shape
    b = torch.cat(
        [a, torch.randn(rc.atom_type_num * 3 - a.shape[0], a.shape[1])], dim=0
    )
    state_dict["decoder.struct_linear.1.weight"] = b
    print(f"{a.shape} => {b.shape}")

    x["state_dict"] = state_dict
    torch.save(x, args.model_files[1])


def modify_ae_model_add_argument(parser):
  parser.add_argument('model_files', type=str, nargs=2, help='list of model files')
  return parser


def modify_latent_model_main(args):
    x = torch.load(args.model_files[0], map_location="cpu", weights_only=False)
    state_dict = x["state_dict"]

    if args.verbose:
        print(state_dict.keys())

    a = state_dict["nn.init_repr_factory.linear_out.weight"]

    if args.model_type == "ucond":
        assert a.shape[1] == 45, a.shape
        b = torch.cat([a, torch.randn(a.shape[0], rc.restype_num - 20)], dim=1)
    elif args.model_type == "motif_idx":
        assert a.shape[1] == 549, a.shape
        # [3, 8, 3, 8, 3, 20, 494] => [3, 8, 3, 8, 3, 30, 854]
        # [148, 148, 20, ] => [260, 260, 30, 176, 63, 65]
        b = torch.cat(
            [
                a[:, :45],
                torch.randn(a.shape[0], rc.restype_num - 20),
                a[:, 45:156],
                torch.randn(a.shape[0], (rc.atom_type_num - 37) * 3),
                a[:, 156:193],
                torch.randn(a.shape[0], (rc.atom_type_num - 37) * 1),
                a[:, 193:304],
                torch.randn(a.shape[0], (rc.atom_type_num - 37) * 3),
                a[:, 304:341],
                torch.randn(a.shape[0], (rc.atom_type_num - 37) * 1),
                a[:, 341:361],
                torch.randn(a.shape[0], rc.restype_num - 20),
                a[:, 361:445],
                torch.randn(a.shape[0], (rc.chi_angles_num - 4) * 21),
                a[:, 445:449],
                torch.randn(a.shape[0], (rc.chi_angles_num - 4) * 1),
                a[:, 449:512],
                a[:, 512:549],
                torch.randn(a.shape[0], (rc.atom_type_num - 37) * 1),
            ],
            dim=1
        )
        assert b.shape[1] == 909, b.shape

    print(f"{a.shape} => {b.shape}")
    state_dict["nn.init_repr_factory.linear_out.weight"] = b

    x["state_dict"] = state_dict
    torch.save(x, args.model_files[1])


def modify_latent_model_add_argument(parser):
  parser.add_argument("model_files", type=str, nargs=2, help="list of model files")
  parser.add_argument(
      "-t", "--model_type", type=str, choices=["ucond", "motif_idx", "motif_uidx"], default="ucond"
  )
  return parser


if __name__ == "__main__":
    import argparse

    commands = {
        "modify_ae_model": (modify_ae_model_main, modify_ae_model_add_argument),
        "modify_latent_model": (
            modify_latent_model_main, modify_latent_model_add_argument
        ),
    }

    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest="command", required=True)
    for cmd, (_, add_argument) in commands.items():
        cmd_parser = sub_parsers.add_parser(cmd)
        add_argument(cmd_parser)
        cmd_parser.add_argument(
            "-v", "--verbose", action="store_true", help="verbose"
        )

    args = parser.parse_args()

    work_fn, _ = commands[args.command]
    work_fn(args)
