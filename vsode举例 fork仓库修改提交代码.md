以下是将 MegaTTS3 仓库从 GitHub fork 到你的账户、在 Visual Studio Code (VS Code) 中修改 `tts/gradio_api.py` 和 `tts/infer_cli.py` 文件、添加 `MegaTTS3_Setup_Summary.md` 说明文档，并将更改推送到你的 fork 仓库的详细步骤。步骤将使用中文描述，专注于在 VS Code 中的操作，假设你已经有一个 GitHub 账户，并且在 Mac Studio M1 Max 上安装了 Git 和 VS Code。我们将修改文件以支持 MPS（Metal Performance Shaders），并包含之前生成的 Markdown 文档。步骤基于官方 MegaTTS3 仓库（假设为 `https://github.com/ByteDance/MegaTTS3`），如果你使用其他仓库，请相应调整 URL。

---

### 在 VS Code 中 Fork 和修改 MegaTTS3 仓库的步骤

#### 1. **在 GitHub 上 Fork MegaTTS3 仓库**
1. **登录 GitHub**：
   - 打开浏览器，访问 `https://github.com`，使用你的 GitHub 账户登录。

2. **找到 MegaTTS3 仓库**：
   - 前往官方 MegaTTS3 仓库：`https://github.com/ByteDance/MegaTTS3`。

3. **Fork 仓库**：
   - 点击页面右上角的 **Fork** 按钮。
   - 选择你的 GitHub 账户（例如 `your-username/MegaTTS3`）作为目标。
   - 等待 GitHub 创建你的 fork 仓库（例如 `https://github.com/your-username/MegaTTS3`）。

#### 2. **在 VS Code 中克隆你的 Fork 仓库**
1. **启动 VS Code**：
   - 打开 VS Code 应用程序。

2. **打开终端**：
   - 在 VS Code 中，按 `Ctrl + ~`（或点击菜单栏的 **View > Terminal**）打开内置终端。

3. **克隆仓库**：
   - 在终端中，切换到你希望存储仓库的目录（例如 `~/Projects`）：
     ```bash
     cd ~/Projects
     ```
   - 复制你的 fork 仓库的 HTTPS URL（例如 `https://github.com/your-username/MegaTTS3.git`），在 GitHub 仓库页面点击 **Code** 按钮获取。
   - 克隆仓库：
     ```bash
     git clone https://github.com/your-username/MegaTTS3.git
     cd MegaTTS3
     ```

4. **在 VS Code 中打开项目**：
   - 在 VS Code 中，点击 **File > Open Folder**，选择刚克隆的 `MegaTTS3` 文件夹。
   - 或者在终端中直接打开：
     ```bash
     code .
     ```

5. **验证克隆内容**：
   - 在 VS Code 的 **Explorer** 面板（左侧文件树），检查是否包含 `tts/gradio_api.py`、`tts/infer_cli.py`、`requirements.txt` 等文件。

6. **（可选）添加 Upstream 远程仓库**：
   - 为保持与原始仓库同步，在终端运行：
     ```bash
     git remote add upstream https://github.com/ByteDance/MegaTTS3.git
     ```
   - 验证远程仓库：
     ```bash
     git remote -v
     ```
     - 应显示 `origin`（你的 fork）和 `upstream`（原始仓库）。

#### 3. **创建新分支**
1. **创建分支**：
   - 在 VS Code 中，点击左下角的分支图标（或按 `Ctrl+Shift+~` 打开终端），当前分支应为 `main`。
   - 创建并切换到新分支：
     ```bash
     git checkout -b mps-support
     ```
   - 或者使用 VS Code 的 Source Control 面板：
     - 点击左侧 **Source Control** 图标。
     - 点击分支名称（`main`），选择 **Create Branch**，输入 `mps-support`。

2. **验证分支**：
   - 在终端运行：
     ```bash
     git branch
     ```
     - 应显示 `* mps-support`。
   - 或在 VS Code 左下角确认分支为 `mps-support`。

#### 4. **在 VS Code 中修改 Python 文件**
你需要将 `tts/gradio_api.py` 和 `tts/infer_cli.py` 修改为支持 MPS 的版本。可以手动编辑或替换文件。

##### 方式 1：手动编辑文件
1. **编辑 `tts/gradio_api.py`**：
   - 在 VS Code 的 **Explorer** 面板，找到 `tts/gradio_api.py`，双击打开。
   - 找到 `model_worker` 函数，确保包含 MPS 支持（参考之前的修改）：
     ```python
     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
     print(f"Worker running on device: {device}")
     infer_pipe = MegaTTS3DiTInfer(device=device)
     ```
   - 在 `if __name__ == '__main__':` 块中，确保：
     ```python
     devices = [0] if torch.backends.mps.is_available() else None
     ```
   - 保存文件（`Ctrl+S`）。

2. **编辑 `tts/infer_cli.py`**：
   - 在 **Explorer** 面板，打开 `tts/infer_cli.py`。
   - 修改 `__init__` 方法（约 78-82 行），启用 MPS：
     ```python
     if device is None:
         device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
     else:
         device = torch.device(device)
     self.device = device
     ```
   - 在 `build_model` 方法的末尾（约 163 行，设备分配后），添加调试输出：
     ```python
     print(f"Models loaded on device: {device}")
     ```
   - 确保 `preprocess` 方法（约 176 行）正确使用 `np.pad`：
     ```python
     wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
     ```
   - 保存文件。

##### 方式 2：替换文件
- 如果你有本地修改过的文件（例如 `/Users/mio/MegaTTS3` 中的 `tts/gradio_api.py` 和 `tts/infer_cli.py`）：
  - 在 VS Code 的终端中复制：
    ```bash
    cp /Users/mio/MegaTTS3/tts/gradio_api.py tts/gradio_api.py
    cp /Users/mio/MegaTTS3/tts/infer_cli.py tts/infer_cli.py
    ```
- 或者使用之前的 `infer_cli.py`（artifact_version_id="25461336-571f-4bce-9691-021e03fef107"）。对于 `gradio_api.py`，如果你能提供当前版本，我可以确认具体修改。
- 在 VS Code 中打开替换后的文件，检查 MPS 相关代码（如上）。

3. **验证修改**：
   - 在 VS Code 的 **Source Control** 面板，点击 **Changes** 查看更改。
   - 或在终端运行：
     ```bash
     git diff tts/gradio_api.py tts/infer_cli.py
     ```
     - 确认包含 MPS 支持（`mps`, `torch.device`, 调试输出）。

#### 5. **添加 Markdown 说明文档**
1. **创建 `MegaTTS3_Setup_Summary.md`**：
   - 在 VS Code 的 **Explorer** 面板，右键单击项目根目录，选择 **New File**，命名为 `MegaTTS3_Setup_Summary.md`。
   - 打开文件，复制之前提供的 Markdown 内容（artifact_version_id="c4ebf206-9e4b-4452-a8f2-1f89cce22af9"），包含：
     - Mac Studio M1 Max 的安装步骤。
     - 配置、性能、音频质量、临时文件管理和建议。
   - 粘贴内容，保存文件。

2. **预览 Markdown**：
   - 在 VS Code 中，右键单击 `MegaTTS3_Setup_Summary.md`，选择 **Open Preview**（或按 `Ctrl+Shift+V`）。
   - 确认格式正确（标题、代码块、列表等）。

#### 6. **提交和推送更改**
1. **暂存文件**：
   - 在 VS Code 的 **Source Control** 面板，点击 **Changes** 旁边的 `+` 号，暂存：
     - `tts/gradio_api.py`
     - `tts/infer_cli.py`
     - `MegaTTS3_Setup_Summary.md`
   - 或在终端运行：
     ```bash
     git add tts/gradio_api.py tts/infer_cli.py MegaTTS3_Setup_Summary.md
     ```

2. **提交更改**：
   - 在 **Source Control** 面板，输入提交信息（例如 “添加 Mac Studio M1 Max 的 MPS 支持和设置说明”），点击 **Commit**（勾选图标）。
   - 或在终端：
     ```bash
     git commit -m "添加 Mac Studio M1 Max 的 MPS 支持和设置说明"
     ```

3. **推送到你的 Fork 仓库**：
   - 在 **Source Control** 面板，点击 **... > Push**，推送 `mps-support` 分支。
   - 或在终端：
     ```bash
     git push origin mps-support
     ```

4. **在 GitHub 上验证**：
   - 访问你的 fork 仓库：`https://github.com/your-username/MegaTTS3`。
   - 切换到 `mps-support` 分支。
   - 检查 `tts/gradio_api.py`、`tts/infer_cli.py` 和 `MegaTTS3_Setup_Summary.md` 是否包含你的更改。
   - 点击 `MegaTTS3_Setup_Summary.md`，确认 GitHub 正确渲染 Markdown。

#### 7. **（可选）创建 Pull Request**
如果想将 MPS 支持贡献给原始 MegaTTS3 仓库：
1. **创建 Pull Request**：
   - 在你的 fork 仓库页面，点击 **Pull requests > New pull request**。
   - 设置基础仓库为 `ByteDance/MegaTTS3`，基础分支为 `main`。
   - 设置你的仓库为 `your-username/MegaTTS3`，比较分支为 `mps-support`。
   - 点击 **Create pull request**，添加标题（例如 “添加 Mac Studio M1 Max 的 MPS 支持”）和描述，提及更改和说明文档。

2. **等待审核**：
   - 原始仓库维护者会审阅你的更改，可能需要你根据反馈调整。

#### 8. **（可选）合并到你的 Main 分支**
如果想将更改合并到你的 fork 的 `main` 分支：
1. **本地合并**：
   - 在 VS Code 终端：
     ```bash
     git checkout main
     git merge mps-support
     git push origin main
     ```
2. **通过 GitHub 合并**：
   - 在你的 fork 仓库，创建从 `mps-support` 到 `main` 的 pull request。
   - 通过 GitHub 界面合并。

#### 9. **测试修改后的仓库**
确保你的 fork 仓库正常运行：
1. **激活环境**：
   - 在 VS Code 终端：
     ```bash
     conda activate megatts3-env
     ```

2. **设置 `PYTHONPATH`**：
   ```bash
   export PYTHONPATH="$(pwd):$PYTHONPATH"
   ```

3. **运行 Gradio**：
   ```bash
   python tts/gradio_api.py
   ```
   - 访问 `http://0.0.0.0:7929`。
   - 使用参数：`Infer Timestep=50`, `Intelligibility Weight=1.0`, `Similarity Weight=5.0`。
   - 验证 MPS 使用（终端输出 `Models loaded on device: mps`）和音频质量。

4. **检查临时文件**：
   - 运行后检查：
     ```bash
     ls -l .
     ls -l ~/.gradio/
     ```

### 额外说明
- **VS Code Git 集成**：
  - VS Code 的 **Source Control** 面板简化了 Git 操作（暂存、提交、推送）。
  - 如果不熟悉终端，可以全程使用图形界面：
    - **暂存**：点击 **Changes** 旁的 `+`。
    - **提交**：输入消息，点击勾选图标。
    - **推送**：点击 **... > Push**。

- **Git 配置**：
  - 如果未配置 Git，设置用户信息：
    ```bash
    git config --global user.name "你的名字"
    git config --global user.email "你的邮箱@example.com"
    ```
  - 如果推送时遇到认证问题：
    - 在 GitHub 生成 Personal Access Token（`https://github.com/settings/tokens`）。
    - 或配置 SSH：
      ```bash
      ssh-keygen -t ed25519
      cat ~/.ssh/id_ed25519.pub
      ```
      - 将公钥添加到 GitHub 的 SSH 设置。

- **文件备份**：
  - 修改前备份原始文件：
    ```bash
    cp tts/gradio_api.py tts/gradio_api.py.bak
    cp tts/infer_cli.py tts/infer_cli.py.bak
    ```

- **与 Upstream 同步**（可选）：
   - 更新你的 fork 与原始仓库：
     ```bash
     git fetch upstream
     git checkout main
     git merge upstream/main
     git push origin main
     ```

- **可能的问题**：
  - **冲突**：如果原始仓库的文件与你的更改冲突，手动解决冲突（VS Code 会高亮冲突部分）。
  - **依赖**：测试 fork 时，确保环境正确：
    ```bash
    conda activate megatts3-env
    pip install -r requirements.txt
    conda install -y -c conda-forge pynini==2.1.5
    pip install WeTextProcessing==1.0.3 pydantic==2.5.0 gradio==5.23.3
    conda install -c conda-forge ffmpeg
    ```

- **Markdown 自定义**：
  - 如果需要调整 `MegaTTS3_Setup_Summary.md`（例如添加截图），在 VS Code 中编辑并预览。

### 总结
通过以上步骤，你可以在 VS Code 中：
1. Fork `ByteDance/MegaTTS3` 到 `your-username/MegaTTS3`。
2. 克隆 fork 仓库，创建 `mps-support` 分支。
3. 修改 `tts/gradio_api.py` 和 `tts/infer_cli.py` 以支持 MPS。
4. 添加 `MegaTTS3_Setup_Summary.md` 文档。
5. 提交并推送更改到你的 fork 仓库。

这些更改使 MegaTTS3 在 Mac Studio M1 Max 上高效运行，说明文档清晰记录了过程。

祝你在 VS Code 和 GitHub 上操作顺利！🚀