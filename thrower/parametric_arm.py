import xacro
import pybullet as pybullet
import numpy as np
import pinocchio
import sys, os

class XacroGenerator():

    def __init__(self, header=None, definition=None, tail=None):
        if header is not None:
            self.header = open(header, mode='r').readlines()
        if definition is not None:
            self.definition = open(definition, mode='r').readlines()
        if tail is not None:
            self.tail = open(tail, mode='r').readlines()

    def __repr__(self):
        out = ''
        try:
            for line in  self.header + self.definition + self.tail:
                out += line
        except Exception as e:
            out += 'Unintialized generator'
        return out

    def _writeModelParameters(self, file, params):
        file.write('\n')
        for tag in params.keys():
            file.write(f'<xacro:property name="{tag}" value="{params[tag]}"/>\n')

    def _writeModelParametersHeader(self, file, params):
        file.write('\n')
        file.write(f'<!-- BEGIN MODEL VARIABLES LIST -->\n')
        for tag in params.keys():
            file.write(f'<!-- name="{tag}" value="{params[tag]}" -->\n')
        file.write(f'<!-- END MODEL VARIABLES LIST -->\n')

    def _writeModel(self, file):
        file.writelines(self.definition)

    def buildXacroModel(self, xacroFilePath, params, writeModelParametersHeader=False):
        xacro_file = open(xacroFilePath, mode='w')
        xacro_file.writelines(self.header)
        if writeModelParametersHeader:
            self._writeModelParametersHeader(xacro_file, params)
        self._writeModelParameters(xacro_file, params)
        self._writeModel(xacro_file)
        xacro_file.writelines(self.tail)
        xacro_file.close()

    def buildURDFModel(self, xacroFilePath, urdfFilePath):
        self.urdf = urdfFilePath
        doc = xacro.process_file(xacroFilePath)
        out_file = xacro.open_output(urdfFilePath)
        out_file.write(doc.toprettyxml(indent='   '))
        out_file.close()

    def returnModel(self, root_joint=pinocchio.JointModelFreeFlyer()):
        return pinocchio.buildModelFromUrdf(self.urdf, root_joint) #self.urdf


path = os.path.abspath(__file__).replace(__file__, "").split("/xacro_modeling")[0]
header = path + "./thrower/model/xacro/header_arm.xml"
definition = path + "./thrower/model/xacro/definition_arm.xml"
tail = path + "./thrower/model/xacro/tail_arm.xml"

testModel = XacroGenerator(header, definition, tail)
xacroOutPath = "/tmp/tmpArm.xml"
urdfOutPath = path + "./model/parametricArm.urdf"

# Defining the default parameters
params = {
    "link_0": 0.1,
    "link_1": 0.1,
}

def build(params):
    testModel.buildXacroModel(xacroOutPath, params)
    testModel.buildURDFModel(xacroOutPath, urdfOutPath)

def build_to(params, xacro_out, urdf_out):
    """Like build(), but writes to caller-specified paths (safe for parallel use)."""
    testModel.buildXacroModel(xacro_out, params)
    testModel.buildURDFModel(xacro_out, urdf_out)

if __name__ == "__main__":

    build(params)

    from geco.design.checkModel import visualizeMeshcat
    from geco.design.checkModel import showModelDetails

    viz = visualizeMeshcat(urdfOutPath, )
    # out = showModelDetails(viz.model)
    # print(out)

    from geco.utils.meshcat_utils import ForceDraw
    fd = ForceDraw(viz.viewer, viz.model, viz.data)
    fd.draw_frames(pinocchio.neutral(viz.model))