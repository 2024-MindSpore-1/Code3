ResNet_source=(['    def construct(self, x):\n', '        if self.use_se:\n', '            x = self.conv1_0(x)\n', '            x = self.bn1_0(x)\n', '            x = self.relu(x)\n', '            x = self.conv1_1(x)\n', '            x = self.bn1_1(x)\n', '            x = self.relu(x)\n', '            x = self.conv1_2(x)\n', '        else:\n', '            x = self.conv1(x)\n', '        x = self.bn1(x)\n', '        x = self.relu(x)\n', '        if self.res_base:\n', '            x = self.pad(x)\n', '        c1 = self.maxpool(x)\n', '\n', '        c2 = self.layer1(c1)\n', '        c3 = self.layer2(c2)\n', '        c4 = self.layer3(c3)\n', '        c5 = self.layer4(c4)\n', '\n', '        out = self.mean(c5, (2, 3))\n', '        out = self.flatten(out)\n', '        out = self.end_point(out)\n', '\n', '        return out\n'],475)
ResidualBlock_source=(['    def construct(self, x):\n', '        identity = x\n', '\n', '        out = self.conv1(x)\n', '        out = self.bn1(out)\n', '        out = self.relu(out)\n', '        if self.use_se and self.stride != 1:\n', '            out = self.e2(out)\n', '        else:\n', '            out = self.conv2(out)\n', '            out = self.bn2(out)\n', '            out = self.relu(out)\n', '        out = self.conv3(out)\n', '        out = self.bn3(out)\n', '        if self.se_block:\n', '            out_se = out\n', '            out = self.se_global_pool(out, (2, 3))\n', '            out = self.se_dense_0(out)\n', '            out = self.relu(out)\n', '            out = self.se_dense_1(out)\n', '            out = self.se_sigmoid(out)\n', '            out = ops.reshape(out, ops.shape(out) + (1, 1))\n', '            out = self.se_mul(out, out_se)\n', '\n', '        if self.down_sample:\n', '            identity = self.down_sample_layer(identity)\n', '\n', '        out = out + identity\n', '        out = self.relu(out)\n', '\n', '        return out\n'],252)
