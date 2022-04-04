from rmil.rmil import main, get_transforms, get_options
from rmil.rmil import get_mil_data, get_aux_data, get_ssl_data


if __name__ == '__main__':
    args = get_options()
    print(args)

    #########################################
    # data
    transform_train, transform_test = get_transforms(args)
    dataloaders_mil = get_mil_data(args, transform_train, transform_test)

    dataloaders_aux, dataloaders_ssl = None, None
    if args.aux or args.reg:
        dataloaders_aux = get_aux_data(args, transform_train, transform_test)
    if args.ssl:
        dataloaders_ssl = get_ssl_data(args, transform_train, transform_test)

    main(args, dataloaders_mil, dataloaders_aux, dataloaders_ssl)
