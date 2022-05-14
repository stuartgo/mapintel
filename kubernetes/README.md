## Creating an AWS EBS volume

Before you can use an EBS volume with a pod, you need to create it.

`aws ec2 create-volume --availability-zone=eu-west-1a --size=10 --volume-type=gp2`

Make sure the zone matches the zone you brought up your cluster in. Check that the size and EBS volume type are suitable for your use.
