---
title: Home
layout: home
---

This is a *bare-minimum* template to create a Jekyll site that uses the [Just the Docs] theme. You can easily set the created site to be published on [GitHub Pages] – the [README] file explains how to do that, along with other details.

If [Jekyll] is installed on your computer, you can also build and preview the created site *locally*. This lets you test changes before committing them, and avoids waiting for GitHub Pages.[^1] And you will be able to deploy your local build to a different platform than GitHub Pages.

More specifically, the created site:

- uses a gem-based approach, i.e. uses a `Gemfile` and loads the `just-the-docs` gem
- uses the [GitHub Pages / Actions workflow] to build and publish the site on GitHub Pages

Other than that, you're free to customize sites that you create with this template, however you like. You can easily change the versions of `just-the-docs` and Jekyll it uses, as well as adding further plugins.

[Browse our documentation][Just the Docs] to learn more about how to use this theme.

To get started with creating a site, simply:

1. click "[use this template]" to create a GitHub repository
2. go to Settings > Pages > Build and deployment > Source, and select GitHub Actions

   # 实验步骤

### 实验2.1 VLAN配置步骤及具体命令  

**实验环境**：华为三层交换机（以用户提供的端口信息为例）  

---

#### **1. 创建VLAN并命名**  

- **创建VLAN 2和VLAN 3**：  

  ```  
  <Switch> system-view  
  [Switch] vlan batch 2 3  
  ```

---

#### **2. 将端口分配到VLAN**  

- **PC1连接端口`Ethernet 0/0/7`分配至VLAN2**：  

  ```  
  [Switch] interface Ethernet 0/0/7  
  [Switch-Ethernet0/0/7] port link-type access  
  [Switch-Ethernet0/0/7] port default vlan 2  
  [Switch-Ethernet0/0/7] quit  
  ```

- **PC2连接端口`Ethernet 0/0/10`分配至VLAN3**：  

  ```  
  [Switch] interface Ethernet 0/0/10  
  [Switch-Ethernet0/0/10] port link-type access  
  [Switch-Ethernet0/0/10] port default vlan 3  
  [Switch-Ethernet0/0/10] quit  
  ```

---

#### **3. 配置VLAN接口的IPv6地址（三层功能）**  

- **为VLAN2和VLAN3配置IPv6地址**：  

  ```  
  [Switch] interface Vlanif 2  
  [Switch-Vlanif2] ipv6 enable  
  [Switch-Vlanif2] ipv6 address 2001:db8:3::1/64  
  [Switch-Vlanif2] quit  
  
  [Switch] interface Vlanif 3  
  [Switch-Vlanif3] ipv6 enable  
  [Switch-Vlanif3] ipv6 address 2001:db8:1::2/64  
  [Switch-Vlanif3] quit  
  ```

---

#### **4. 验证配置**  

- **查看VLAN信息**：  

  ```  
  [Switch] display vlan  
  ```

  输出应显示：  

  - VLAN2包含端口`Ethernet0/0/7`  
  - VLAN3包含端口`Ethernet0/0/10`  

- **查看接口IPv6配置**：  

  ```  
  [Switch] display ipv6 interface Vlanif 2  
  [Switch] display ipv6 interface Vlanif 3  
  ```

---

#### **5. PC配置**  

- **PC1（VLAN2）**：  
  - IPv6地址：`2001:db8:1::1/64`  
  - 网关：`2001:db8:3::1`  
- **PC2（VLAN3）**：  
  - IPv6地址：`2001:db8:1::2/64`  
  - 网关：`2001:db8:2::1`  

---

#### **6. 连通性测试**  

- **同一VLAN内（PC1 ping网关）**：  

  ```  
  C:\> ping 2001:db8:1::1  
  ```

  **预期结果**：成功（Reply from 2001:db8:1::1）。  

- **不同VLAN间（PC1 ping PC2）**：  

  ```  
  C:\> ping 2001:db8:2::2  
  ```

  **预期结果**：成功（需三层交换机启用IPv6路由）。  

---

#### **7. 关键问题解答**  

- **默认VLAN数量**：  
  通过`display vlan`查看，默认VLAN为VLAN1，包含所有未分配的端口（如未使用的端口）。  

- **验证VLAN端口分配**：  
  使用`display vlan`确认`Ethernet0/0/7`属于VLAN2，`Ethernet0/0/10`属于VLAN3。  

- **MAC地址表查看**：  

  ```  
  [Switch] display mac-address  
  ```

  输出字段包括：`MAC地址、VLAN、端口、类型（动态/静态）`。  

---

#### **命令总结表**  

| 步骤             | 命令示例                                                     |
| ---------------- | ------------------------------------------------------------ |
| 创建VLAN         | `vlan batch 2 3`                                             |
| 命名VLAN         | `vlan 2` → `name VLAN2`                                      |
| 分配端口至VLAN2  | `interface Ethernet0/0/7` → `port link-type access` → `port default vlan 2` |
| 分配端口至VLAN3  | `interface Ethernet0/0/10` → `port link-type access` → `port default vlan 3` |
| 配置VLAN接口IPv6 | `interface Vlanif 2` → `ipv6 enable` → `ipv6 address 2001:db8:1::1/64` |
| 验证配置         | `display vlan`、`display ipv6 interface brief`               |

---

#### **注意事项**  

1. 确保交换机已启用IPv6路由功能（默认可能未启用）：  

   ```  
   [Switch] ipv6  
   ```

2. 若无法跨VLAN通信，检查：  

   - VLAN接口IPv6地址是否正确  
   - PC网关是否配置正确  
   - 防火墙或ACL是否拦截流量  

3. 保存配置防止丢失：  

   ```  
   [Switch] save  
   ```

If you want to maintain your docs in the `docs` directory of an existing project repo, see [Hosting your docs from an existing project repo](https://github.com/just-the-docs/just-the-docs-template/blob/main/README.md#hosting-your-docs-from-an-existing-project-repo) in the template README.

----

[^1]: [It can take up to 10 minutes for changes to your site to publish after you push the changes to GitHub](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll#creating-your-site).

[Just the Docs]: https://just-the-docs.github.io/just-the-docs/
[GitHub Pages]: https://docs.github.com/en/pages
[README]: https://github.com/just-the-docs/just-the-docs-template/blob/main/README.md
[Jekyll]: https://jekyllrb.com
[GitHub Pages / Actions workflow]: https://github.blog/changelog/2022-07-27-github-pages-custom-github-actions-workflows-beta/
[use this template]: https://github.com/just-the-docs/just-the-docs-template/generate
