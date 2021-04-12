from OHT.BasicClientActor import BasicClientActor


def play_OHT():
    client = BasicClientActor(IP_address="129.241.231.193")
    client.connect_to_server()


if __name__ == "__main__":
    play_OHT()
