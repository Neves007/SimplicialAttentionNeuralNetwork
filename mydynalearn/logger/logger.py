from loguru import logger as log
import numpy as np
import os
class logger():
    def __init__(self,config):
        self.fileName = 'logFile.log'
        self.logFilePath = os.path.join(config.homePath, self.fileName)

        format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> |-- <level>{message}</level>'
        log.add(self.logFilePath,mode='w',format=format)
    def generate_data(self,network,dynamics):
        log.warning("Successfully generated data!")
        log.info("")
        netInfoHead = 30*"*"+" Network "+30*"*"
        log.info(netInfoHead)
        log.info("Netowrk type:\t {}".format(network.name))
        log.info("Number of nodes:\t {:d}".format(network.num_nodes))
        log.info("Network simplex max dimension:\t {:d}".format(network.maxDimension))
        log.info("Network mean degree:\t "+(str(network.avg_k)))
        log.info(len(netInfoHead) * "*" )
        log.info("")
        dynamicInfoHead = 30 * "*" + " Dynamic " + 30 * "*"
        log.info(dynamicInfoHead)
        log.info("Dynamic type:\t {}".format(dynamics.name))
        log.info("Dynamic simplex max dimension:\t {:d}".format(dynamics.maxDimension))
        log.info("Effective Infection probability:\t "+str(dynamics.eff_infection.numpy()))
        log.info("recovery probability:\t {}".format(dynamics.mu))
        log.info(len(dynamicInfoHead) * "*")
        log.info("")
        log.info("")
    def TestEpoch(self,epoch_index):
        log.success("Successfully test epoch {:d}!".format(epoch_index))
    def TrainEpoch(self,epoch_index):
        log.success("Successfully train epoch {:d}!".format(epoch_index))
    def evaluate(self):
        log.success("Successfully draw evaluation figures!")