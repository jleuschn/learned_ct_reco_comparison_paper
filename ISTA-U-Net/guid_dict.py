# -*- coding: utf-8 -*-

guid_dict = {
    'noisefree': {
        50: '818dea8a-24a6-4a5b-811f-65c38c9ecc61',  # epochs 20
        10: '17281e91-ba0e-4ad4-9c5c-97673b9ba324',  # epochs 20
        5: '19ffea0a-c387-48b8-a445-08ff77c3b22d',  # epochs 20
        2: 'f4771152-3a8b-4742-ab51-d1830915382a'  # epochs 10, best val psnr after 3 epochs
        # 50: '0cd136af-66ff-4903-9c7e-03bf07255039'  # epochs 5: 37.96dB, sub-optimal
        # 2: '4af56170-2c92-43df-943d-691de2150f1c'  # epochs 5
        # 2: 'b6a6ac87-3e70-4343-badd-dea8cc9988e6'  # 
    },
    'gaussian_noise': {
        50: 'b3e4f75d-0a92-4843-b4b8-4ec4afe9e5e1',  # epochs 20
        # 50: '52e07ea6-fc7f-417e-a90d-52caeb49ee1f'  # old, widths [512, 256, 128, 64, 32], init lambda 1e-3, 15 epochs: 36.19dB, nnz ['0.00', '0.02', '0.01', '0.08', '0.00']
        10: 'c63763ab-88cb-490c-b9d9-fb676accb661',  # epochs 20
        # 10: '77b1daf8-5b0a-4207-9470-8aa49f5ec4d9',  # epochs 20, best val psnr after 17 epochs, but weights were not stored (<0.1 dB better than val psnr after 20 epochs)
        # 10: '0818989e-1451-468f-bfad-fd859851fc6f',  # retry, epochs 20, crashed after 9 epochs
        5: 'dc22ca7f-a6db-457f-8bfb-9c4b48cf19a8',  # epochs 20
        # 5: '9cfcb7d2-103c-4a83-a5f7-69c17ccc8d5e',  # <0.2dB better than 'dc22ca7f-a6db-457f-8bfb-9c4b48cf19a8', hyper parameters are identical, missing validation psnrs of epochs 9, 10 (did not copy while in screen buffer)
        2: '7c7e82b8-4e08-4797-97a5-190fab47bdcd'  # retry, epochs 10, crashed after 9 epochs, best val psnr after 4 epochs
        # 2: '817e33ea-e335-49bf-a40f-f098847d717d'  # best val psnr after 5 epochs, slightly better than for '7c7e82b8-4e08-4797-97a5-190fab47bdcd' (<0.1dB), but weights were not stored
        # 2: 'cecb5375-7a7f-49cb-a5b2-66ba2015700a'  # 5 epochs
    },
    'scattering': {
        50: '1a72ac72-b8d4-45a9-8d71-9e4d8cae6e98',  # epochs 80
        # 50: 'f6626a49-26a2-47ef-a938-40d160d3e757',  # 
        # 50: '65820ef7-a034-4ebc-968b-13f31c41f85e',  # aborted after 43 epochs due to overfitting
        # 50: '1756c7da-13f8-4e59-915d-6062e388413f',  # epochs 20, 33.96dB, nnz ['0.58', '0.65', '0.56', '0.67', '0.57', '0.62']
        10: 'b465d2fe-0ae7-426f-b134-0e12ca80e8b9',  # epochs 40
        # 10: 'bab6f8c0-6174-45e0-8090-b0a2143575c6',  # epochs 20, 31.14dB, nnz ['0.45', '0.44', '0.55', '0.52', '0.53', '0.62']
        5: '34fe1fae-48a6-410d-813a-56e8214ee31e',  # epochs 40
        # 5: 'f6b01af6-3053-467b-90c5-ac6498415fd2',  # epochs 20: 26.30dB, nnz ['0.58', '0.55', '0.58', '0.52', '0.51', '0.63']  (could maybe use some more epochs)
        2: '909a8e1c-c2e5-435c-82b6-d97664641801'  # epochs 20, best val psnr after 7 epochs
        # 2: '6f36b94f-5b74-4b6f-bc80-acb6c4769553'  # epochs 15: 19.31dB, nnz ['0.29', '0.25', '0.31', '0.42', '0.51', '0.53']  (overfitted slightly, judging from latest validation PSNRs)
        # 2: '56dac93b-f504-426d-b7f0-f778e90ad88d'  # retry, epochs 40
        # 2: '77ccb4b6-0622-4c42-a15a-4c1630525795'  # retry, epochs 10
    }
}
