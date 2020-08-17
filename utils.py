def update_target(tar_net, net, update_rate):
    for tar_net_paras, net_paras in zip(tar_net.parameters(), net.parameters()):
        tar_net_paras.data.copy_((1 - update_rate) * tar_net_paras
                                 + update_rate * net_paras)
