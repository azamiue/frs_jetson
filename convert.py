# pnet = PNet()
# pnet.load_state_dict(torch.load(r'C:\Users\thainq\Desktop\capstone\frs_jetson\mtcnn\data\pnet.pt'))
# pnet.eval()
# scripted_pnet = torch.jit.script(pnet)
# print("pnet:", pnet)
# print("scripted_pnet:", scripted_pnet)
# torch.jit.save(scripted_pnet, 'scripted_pnet.pt')

# scripted_pnet = torch.jit.load("scripted_pnet.pt")
# print("scripted_pnet:", scripted_pnet)